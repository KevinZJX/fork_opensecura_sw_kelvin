/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

// Convolution based on Kelvin ops
// Data types: input: s8, filter: s8, bias: s32
// Special case for 48x3x1x48 filter

#include "tflm/opt/conv_util.h"

namespace kelvin::opt {

void ConvS8K3x1D48(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  const auto batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int32_t input_offset = params.input_offset;
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_depth = filter_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_depth = output_shape.Dims(3);
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  TFLITE_DCHECK(batches == 1);
  TFLITE_DCHECK(filter_depth == input_depth);
  TFLITE_DCHECK(filter_height == 3);
  TFLITE_DCHECK(filter_width == 1);
  TFLITE_DCHECK(input_width == 1);
  TFLITE_DCHECK(stride_width == 1);
  TFLITE_DCHECK(stride_height == 1);
  TFLITE_DCHECK(dilation_width_factor == 1);
  TFLITE_DCHECK(dilation_height_factor == 1);
  TFLITE_DCHECK(pad_width == 0);
  TFLITE_DCHECK(pad_height == 0);

  int32_t bias[48 * 4];
  int32_t mult[48 * 4];
  int32_t shft[48 * 4];
  Swizzle(bias_data, bias, 48);
  Swizzle(output_multiplier, mult, 48);
  Swizzle(output_shift, shft, 48, true);

  int8_t juggled_filter_data[48 / 8][3][1][48 / 4][8][4];
  Filter_N_H_W_M<48>(filter_data, juggled_filter_data[0][0][0][0][0], 3, 1, 48);
  union {
    vconv_u8_t conv;
    uint32_t raw;
  } cmds;
  cmds.conv.mode = 0;
  cmds.conv.start = 0;
  cmds.conv.stop = 7;
  cmds.conv.sbias1 = input_offset;
  cmds.conv.sdata1 = true;
  cmds.conv.sbias2 = 0;
  cmds.conv.sdata2 = true;

  union {
    vconv_u8_t conv;
    uint32_t raw;
  } cmds16;
  cmds16.conv.mode = 0;
  cmds16.conv.start = 0;
  cmds16.conv.stop = 3;
  cmds16.conv.sbias1 = input_offset;
  cmds16.conv.sdata1 = true;
  cmds16.conv.sbias2 = 0;
  cmds16.conv.sdata2 = true;

  for (int zo_hi = 0; zo_hi < output_depth; zo_hi += 8) {
// For each pixel, the general flow for this kernel looks like:
// 1) Reset accumulator and load activations into [v32, v46]
// 2) For each group of 32 scalars in the pixel fan-in, run MAC pipeline
//    2a) Load subset of activations from [v32, v46] to [v0, v7]
//    2b) Load subset of weights
//    2c) Run aconv
// 3) Run the output pipeline and store.
//
// For step 1, we'll alias [v32, v46] to [L0, LE]. For most iterations,
// we load all of these registers (10 pixels). For remainder iterations,
// we load a subset and pad the rest with 0's. The data will be stored as
// follows, where each letter represents 16 bytes of a pixel stored into
// a register (capitalization used to help distinguish channels in a pixel):
// L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 LA LB LC LD LE
// Aa AB bB Cc CD dD Ee EF fF Gg GH hH Ii IJ jJ
#define L0 v32
#define L1 v33
#define L2 v34
#define L3 v35
#define L4 v36
#define L5 v37
#define L6 v38
#define L7 v39
#define L8 v40
#define L9 v41
#define LA v42
#define LB v43
#define LC v44
#define LD v45
#define LE v46

// We run 5 iterations of step 2, 4 full iterations and one half iteration.
// Because each pixel takes 1.5 registers, we have to interleave vmv_v and
// vsliden_w_4_vv instructions to ensure the same output channels are stored
// in each register per-pixel. As a refresher, vsliden_w_4_vv takes two
// register arguments (X and Y), and returns the concatenation of the last
// half of X and the first half of Y. ie:
// L1 L2
// AB bB
// vsliden_w_4_vv(v1, L1, L2); -> v1 = Bb
#define CONV_PER_CHANNEL_B8_3X1_48C_MAC_PIPELINE(p_flt)              \
  {                                                                  \
    /* 1/5 */                                                        \
    /* Ky = 0, IC:[0-31] */                                          \
    vmv_v(v0, L0);              /* Aa */                             \
    vsliden_w_4_vv(v1, L1, L2); /* Bb */                             \
    vmv_v(v2, L3);              /* Cc */                             \
    vsliden_w_4_vv(v3, L4, L5); /* Dd */                             \
    vmv_v(v4, L6);              /* Ee */                             \
    vsliden_w_4_vv(v5, L7, L8); /* Ff */                             \
    vmv_v(v6, L9);              /* Gg */                             \
    vsliden_w_4_vv(v7, LA, LB); /* Hh */                             \
    vld_b_x_m(v56, p_flt + 128 * 0);                                 \
    vld_b_x_m(v60, p_flt + 128 * 1);                                 \
    aconv_vxv(v48, v0, cmds, v56);                                   \
                                                                     \
    /* 2/5 */                                                        \
    /* Ky = 0, IC:[32-47]; Ky = 1, IC:[0-15] */                      \
    vmv_v(v0, L1);              /* AB */                             \
    vsliden_w_4_vv(v1, L2, L3); /* BC */                             \
    vmv_v(v2, L4);              /* CD */                             \
    vsliden_w_4_vv(v3, L5, L6); /* DE */                             \
    vmv_v(v4, L7);              /* EF */                             \
    vsliden_w_4_vv(v5, L8, L9); /* FG */                             \
    vmv_v(v6, LA);              /* GH */                             \
    vsliden_w_4_vv(v7, LB, LC); /* HI */                             \
    vld_b_x_m(v56, p_flt + 128 * 2);                                 \
    vld_b_x_m(v60, p_flt + 128 * 3);                                 \
    aconv_vxv(v48, v0, cmds, v56);                                   \
                                                                     \
    /* 3/5 */                                                        \
    /* Ky = 1, IC:[16-47] */                                         \
    vmv_v(v0, L2);              /* bB */                             \
    vsliden_w_4_vv(v1, L3, L4); /* cC */                             \
    vmv_v(v2, L5);              /* dD */                             \
    vsliden_w_4_vv(v3, L6, L7); /* eE */                             \
    vmv_v(v4, L8);              /* fF */                             \
    vsliden_w_4_vv(v5, L9, LA); /* gG */                             \
    vmv_v(v6, LB);              /* hH */                             \
    vsliden_w_4_vv(v7, LC, LD); /* iI */                             \
    vld_b_x_m(v56, p_flt + 128 * 4);                                 \
    vld_b_x_m(v60, p_flt + 128 * 5);                                 \
    aconv_vxv(v48, v0, cmds, v56);                                   \
                                                                     \
    /* 4/5 */                                                        \
    /* Ky = 2, IC:[0-31] */                                          \
    vmv_v(v0, L3);              /* Cc */                             \
    vsliden_w_4_vv(v1, L4, L5); /* Dd */                             \
    vmv_v(v2, L6);              /* Ee */                             \
    vsliden_w_4_vv(v3, L4, L5); /* Ff */                             \
    vmv_v(v4, L9);              /* Gg */                             \
    vsliden_w_4_vv(v5, LA, LB); /* Hh */                             \
    vmv_v(v6, LC);              /* Ii */                             \
    vsliden_w_4_vv(v7, LD, LE); /* Jj */                             \
    vld_b_x_m(v56, p_flt + 128 * 6);                                 \
    vld_b_x_m(v60, p_flt + 128 * 7);                                 \
    aconv_vxv(v48, v0, cmds, v56);                                   \
                                                                     \
    /* 5/5 */                                                        \
    /* Ky = 2, IC:[32-47] half iteration */                          \
    vmv_v(v0, L4);              /* C(D- ignored) */                  \
    vsliden_w_4_vv(v1, L5, L6); /* D(E- ignored) */                  \
    vmv_v(v2, L7);              /* E(F- ignored) */                  \
    vsliden_w_4_vv(v3, L8, L9); /* F(G- ignored) */                  \
    vmv_v(v4, LA);              /* G(H- ignored) */                  \
    vsliden_w_4_vv(v5, LB, LC); /* H(I- ignored) */                  \
    vmv_v(v6, LD);              /* I(J- ignored) */                  \
    /* Pad last iteration with first pixel. Gets ignored by cmd16 */ \
    vsliden_w_4_vv(v7, LE, L0);      /* J(A- ignored) */             \
    vld_b_x_m(v56, p_flt + 128 * 8); /*Load once half iteration*/    \
    /* cmds16 runs subset of outer product */                        \
    aconv_vxv(v48, v0, cmds16, v56);                                 \
  }

    // Iterate over outputs
    int out_y = 0;
    for (; out_y + 8 <= output_height; out_y += 8) {
      // Reset accumulator
      vdup_w_x_m(v48, 0);
      vdup_w_x_m(v52, 0);
      acset_v(v48, v48);

      const int8_t* p_flt = juggled_filter_data[zo_hi / 8][0][0][0][0];
      const int8_t* p_in = input_data + (out_y * input_width * input_depth);

      // Load 10*48 activations into 10*48*32 = 15 registers
      vld_b_x_m(L0, p_in);
      vld_b_x_m(L4, p_in + 32 * 4);
      vld_b_x_m(L8, p_in + 32 * 8);
      vld_b_x(LC, p_in + 32 * 12);
      vld_b_x(LD, p_in + 32 * 13);
      vld_b_x(LE, p_in + 32 * 14);

      // MAC pipeline
      CONV_PER_CHANNEL_B8_3X1_48C_MAC_PIPELINE(p_flt);

      // Output pipeline
      INT32_TO_INT8_OUTPUT_PIPELINE(bias + zo_hi * 4, mult + zo_hi * 4,
                                    shft + zo_hi * 4, output_activation_min,
                                    output_activation_max, output_offset, v36,
                                    v40, v44);
      int8_t* p_out =
          output_data + tflite::Offset(output_shape, 0, out_y, 0, zo_hi);
      vstq_b_sp_xx(v48, p_out, output_depth);
      vstq_b_sp_xx(v52, p_out, output_depth);
    }

    // Left over minibatch
    int remainder = output_height - out_y;
    if (remainder != 0) {
      // Reset accumulator
      vdup_w_x_m(v48, 0);
      vdup_w_x_m(v52, 0);
      acset_v(v48, v48);

      const int8_t* p_flt = juggled_filter_data[zo_hi / 8][0][0][0][0];
      const int8_t* p_in = input_data + (out_y * input_width * input_depth);

      // Load (remainder + 2) * 48 activations
      // L0 L1 L2 L3 L4 L5 L6 L7 L8 L9 LA LB LC LD
      // AA AB BB CC CD DD EE EF FF GG GH HH II I-
      vld_b_x_m(L0, p_in);
      vdup_w_x_m(L4, 0);
      vdup_w_x_m(L8, 0);
      vdup_w_x_m(LC, 0);
      switch (remainder) {
        case 7:
          vld_b_x(LD, p_in + 32 * 13);
          vld_b_x(LC, p_in + 32 * 12);
        case 6:
          vld_b_x(LB, p_in + 32 * 11);
        case 5:
          vld_b_x(LA, p_in + 32 * 10);
          vld_b_x(L9, p_in + 32 * 9);
        case 4:
          vld_b_x(L8, p_in + 32 * 8);
        case 3:
          vld_b_x(L7, p_in + 32 * 7);
          vld_b_x(L6, p_in + 32 * 6);
        case 2:
          vld_b_x(L5, p_in + 32 * 5);
        default:
          break;
      }
      vld_b_x(L4, p_in + 32 * 4);

      // MAC pipeline
      CONV_PER_CHANNEL_B8_3X1_48C_MAC_PIPELINE(p_flt);

      // Output pipeline
      INT32_TO_INT8_OUTPUT_PIPELINE(bias + zo_hi * 4, mult + zo_hi * 4,
                                    shft + zo_hi * 4, output_activation_min,
                                    output_activation_max, output_offset, v36,
                                    v40, v44);

      int8_t* p_out =
          output_data + tflite::Offset(output_shape, 0, out_y, 0, zo_hi);
      uint8_t local_data[64];
      vst_b_x(v0, local_data);
      vst_b_x(v1, local_data + 32);
      for (int i = 0; i < remainder; i++) {
        memcpy(p_out + (i * output_depth), local_data + (i * 8), 8);
      }
    }

#undef CONV_PER_CHANNEL_B8_3X1_48C_MAC_PIPELINE
#undef L0
#undef L1
#undef L2
#undef L3
#undef L4
#undef L5
#undef L6
#undef L7
#undef L8
#undef L9
#undef LA
#undef LB
#undef LC
#undef LD
#undef LE
  }
}

}  // namespace kelvin::opt
