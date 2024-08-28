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

#include <cstdlib>
#include <memory>

#include "crt/kelvin.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tflm/opt/conv_s8.h"
#include "tflm/opt/conv_util.h"
#include "tflm/opt/opt.h"

namespace {

constexpr int32_t kBytesPerRegister = 32;
constexpr int8_t kFilterZeroPoint = 0;
}  // namespace

namespace kelvin::opt {
namespace {

void VectorSwizzle8(const int32_t* input, int32_t* output, int32_t* output2) {
  // swizzle to achive following pattern
  // out 1 : [0, 2, 1, 3, 0, 2, 1, 3]
  // out 2 : [4, 6, 5, 7, 4, 6, 5, 7]

  const int32_t(&in)[8] = *(int32_t(*)[8])input;
  int32_t(&out)[8] = *(int32_t(*)[8])output;
  int32_t(&out2)[8] = *(int32_t(*)[8])output2;

  out[0] = in[0];
  out[2] = in[1];
  out[1] = in[2];
  out[3] = in[3];
  out[4] = in[0];
  out[6] = in[1];
  out[5] = in[2];
  out[7] = in[3];

  out2[0] = in[4];
  out2[2] = in[5];
  out2[1] = in[6];
  out2[3] = in[7];
  out2[4] = in[4];
  out2[6] = in[5];
  out2[5] = in[6];
  out2[7] = in[7];
}

void PaddedFilter(const int8_t* input, int8_t* output, int output_channels) {
  // Filter data is being reorganized into groups of 8 channels and falttening
  // row. 9th element of 3x3 filter is padded (9000 9000 9000 9000) 8 channels
  // are aligned this way     (  c0  c1   c2    c3)

  for (int group = 0; group < output_channels / 8; group++) {
    const int8_t* input_group_pointer = input + (group * 8 * 3 * 3 * 3);
    int8_t* output_group_pointer = output + (group * 8 * 3 * 3 * 4);

    for (int channel = 0; channel < 8; channel++) {
      for (int row = 0; row < 3; row++) {
        int out_row_offset = (channel * 4) + (row * 3 * kBytesPerRegister);

        const int8_t* input_c1_offset =
            input_group_pointer + (channel * 27) + (9 * row);
        int8_t* output_c1_offset = output_group_pointer + out_row_offset;
        memcpy(output_c1_offset, input_c1_offset, 4);

        const int8_t* input_c2_offset =
            input_group_pointer + (channel * 27) + (9 * row + 4);
        int8_t* output_c2_offset =
            output_group_pointer + out_row_offset + kBytesPerRegister;
        memcpy(output_c2_offset, input_c2_offset, 4);

        const int8_t* input_c3_offset =
            input_group_pointer + (channel * 27) + (9 * row + 8);
        int8_t* output_c3_offset =
            output_group_pointer + out_row_offset + (2 * kBytesPerRegister);
        memcpy(output_c3_offset, input_c3_offset, 1);

        *(output_c3_offset + 1) = kFilterZeroPoint;
        *(output_c3_offset + 2) = kFilterZeroPoint;
        *(output_c3_offset + 3) = kFilterZeroPoint;
      }
    }
  }
}
}  // namespace

// IN:
//  - v0, v4 (input pixels)
//  - v11 (zeroed register)
// OUT:
//  - v0, v1, v2, v3, v4, v5, v6, v7 (reforged input columns)
// CLOBBERS:
//  - None
#define FORGING_4_INPUT_COLUMNS_INTO_V0_V7_FOR_ACONV \
  {                                                  \
    vsliden_h_3_vv(v1, v0, v11);                     \
    vsliden_w_3_vv(v2, v0, v11);                     \
    vsliden_w_3_vv(v3, v1, v11);                     \
    vsliden_h_3_vv(v5, v4, v11);                     \
    vsliden_w_3_vv(v6, v4, v11);                     \
    vsliden_w_3_vv(v7, v5, v11);                     \
  }

#define POST_PROCESS_ACONV_INT32_ACC_TO_INT8_ACC  \
  {                                               \
    vadd_w_vv_m(v48, v48, v35);                   \
    vadd_w_vv_m(v52, v52, v35);                   \
    vdmulh_w_rn_vv_m(v48, v48, v12);              \
    vdmulh_w_rn_vv_m(v52, v52, v12);              \
    vsha_w_r_vv_m(v48, v48, v16);                 \
    vsha_w_r_vv_m(v52, v52, v16);                 \
    vadd_w_vx_m(v48, v48, output_offset);         \
    vadd_w_vx_m(v52, v52, output_offset);         \
    vmin_w_vx_m(v48, v48, output_activation_max); \
    vmin_w_vx_m(v52, v52, output_activation_max); \
    vmax_w_vx_m(v48, v48, output_activation_min); \
    vmax_w_vx_m(v52, v52, output_activation_min); \
    vst_w_x_m(v48, &acc_out32[0]);                \
    vst_w_x_m(v52, &acc_out32[32]);               \
    for (int i = 0; i < 4; i++) {                 \
      acc_out8[0][i][0] = acc_out32[i * 16 + 0];  \
      acc_out8[0][i][2] = acc_out32[i * 16 + 1];  \
      acc_out8[0][i][1] = acc_out32[i * 16 + 2];  \
      acc_out8[0][i][3] = acc_out32[i * 16 + 3];  \
      acc_out8[1][i][0] = acc_out32[i * 16 + 4];  \
      acc_out8[1][i][2] = acc_out32[i * 16 + 5];  \
      acc_out8[1][i][1] = acc_out32[i * 16 + 6];  \
      acc_out8[1][i][3] = acc_out32[i * 16 + 7];  \
      acc_out8[0][i][4] = acc_out32[i * 16 + 8];  \
      acc_out8[0][i][6] = acc_out32[i * 16 + 9];  \
      acc_out8[0][i][5] = acc_out32[i * 16 + 10]; \
      acc_out8[0][i][7] = acc_out32[i * 16 + 11]; \
      acc_out8[1][i][4] = acc_out32[i * 16 + 12]; \
      acc_out8[1][i][6] = acc_out32[i * 16 + 13]; \
      acc_out8[1][i][5] = acc_out32[i * 16 + 14]; \
      acc_out8[1][i][7] = acc_out32[i * 16 + 15]; \
    }                                             \
  }

void ConvS8I3xD8(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;       // r = s(q - Z)
  const int32_t neg_input_offset = -params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.Dims(2), 3);
  TFLITE_DCHECK_EQ(filter_shape.Dims(1), 3);
  TFLITE_DCHECK_EQ(input_shape.Dims(3), 3);
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_NE(groups, 0);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  TFLITE_DCHECK_EQ(filter_input_depth, 3);
  const int filters_per_group = output_depth / groups;
  TFLITE_DCHECK_NE(filters_per_group, 0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  union {
    vconv_u8_t conv;
    uint32_t raw;
  } cmds;
  cmds.conv.mode = 0;
  cmds.conv.start = 0;
  cmds.conv.stop = 2;
  cmds.conv.sbias1 = input_offset;
  cmds.conv.sdata1 = true;
  cmds.conv.sbias2 = 0;
  cmds.conv.sdata2 = true;

  // Reg Map:
  // v0-v7   : input patches[0..7]
  // v8-v10  : filter row 1 (registers used for aconv)
  // v11     : vdup 0 used during vsliden
  // v12-v15 : Swizzled Biases
  // v16-v19 : Swizzled shift mulitpliers
  // v24-v26 : filter row 2 (registers used for aconv)
  // v30     : negative offset mask
  // v34-v37 : Swizzled Biases
  // v40-v42 : filter row 3 (registers used for aconv)
  // v48-v55 : Accumulators for aconv

  int8_t acc_out8[2][4][8];
  int32_t acc_out32[64];
  int out_channel = 0;
  const size_t swizzled_filter_data_size = output_depth * 3 * 3 * 4;
  std::unique_ptr<int8_t> swizzled_filter_data(reinterpret_cast<int8_t*>(
      ::aligned_alloc(32, swizzled_filter_data_size)));
  int8_t* p_swizzled_filter_data = swizzled_filter_data.get();

  PaddedFilter(filter_data, p_swizzled_filter_data, output_depth);
  // structure of padded filter data : 1st row 0-8 channels 0-95 , 2nd row 0-8
  // channels 96-191, 3rd row 0-8 channels 192-287

  do {
    int32_t temp_data_shuffle[2][8]{0};

    VectorSwizzle8(bias_data + out_channel, &temp_data_shuffle[0][0],
                   &temp_data_shuffle[1][0]);
    vld_w_x(v35, &temp_data_shuffle[0][0]);
    vld_w_x(v36, &temp_data_shuffle[1][0]);
    vmv_v(v37, v35);
    vmv_v(v38, v36);

    VectorSwizzle8(output_multiplier + out_channel, &temp_data_shuffle[0][0],
                   &temp_data_shuffle[1][0]);
    vld_w_x(v12, &temp_data_shuffle[0][0]);
    vld_w_x(v13, &temp_data_shuffle[1][0]);
    vmv_v(v14, v12);
    vmv_v(v15, v13);

    VectorSwizzle8(output_shift + out_channel, &temp_data_shuffle[0][0],
                   &temp_data_shuffle[1][0]);
    vld_w_x(v16, &temp_data_shuffle[0][0]);
    vld_w_x(v17, &temp_data_shuffle[1][0]);
    vmv_v(v18, v16);
    vmv_v(v19, v17);
    vrsub_w_vx_m(v16, v16, 0);
    vdup_b_x(v11, 0);  // used for vsliden

    int8_t mask[32] = {0};
    for (int i = 24; i < 32; ++i) {
      mask[i] = neg_input_offset;
    }
    vld_b_x(v30, mask);  // mast to negate input offset

    int fil_channels_offset = out_channel / 8;

    // load filter this is done once per change in 8 channels
    vld_b_x(v8, p_swizzled_filter_data + fil_channels_offset * 288);  // row 1
    vld_b_x(v9,
            p_swizzled_filter_data + fil_channels_offset * 288 + 32);  // row 1
    vld_b_x(v10,
            p_swizzled_filter_data + fil_channels_offset * 288 + 64);  // row 1

    vld_b_x(v24,
            p_swizzled_filter_data + fil_channels_offset * 288 + 96);  // row 2
    vld_b_x(v25,
            p_swizzled_filter_data + fil_channels_offset * 288 + 128);  // row 2
    vld_b_x(v26,
            p_swizzled_filter_data + fil_channels_offset * 288 + 160);  // row 2

    vld_b_x(v40,
            p_swizzled_filter_data + fil_channels_offset * 288 + 192);  // row 3
    vld_b_x(v41,
            p_swizzled_filter_data + fil_channels_offset * 288 + 224);  // row 3
    vld_b_x(v42,
            p_swizzled_filter_data + fil_channels_offset * 288 + 256);  // row 3

    for (int batch = 0; batch < batches; ++batch) {
      int8_t* p_output = output_data +
                         (batch * output_height * output_width * output_depth) +
                         out_channel;
      const int8_t* p_input =
          input_data + (batch * input_height * input_width * input_depth);
      for (int out_y = 0; out_y + 2 < output_height; out_y += 2) {
        const int in_y_origin = (out_y * stride_height);
        for (int out_x = 0; out_x + 4 < output_width; out_x += 4) {
          const int in_x_origin = (out_x * stride_width);

          vdup_w_x_m(v48, 0);
          vdup_w_x_m(v52, 0);

          acset_v(v48, v48);

          // inputs row 0 and row 2
          vld_b_x(v0, p_input + (in_y_origin * input_width * input_depth) +
                          (in_x_origin * input_depth));
          vld_b_x(v4, p_input +
                          ((in_y_origin + 2) * input_width * input_depth) +
                          (in_x_origin * input_depth));

          // explaining data slide strategy
          // v0 loads 10 RGB pixels of Row 0 ( which turn out to be first 30
          // values) 0 1 2 3 4 5 6 7 8 9 first vslide corresponds to outx + 1 2
          // 3 4 5 6 7 8 9 ( stride == 2) 2nd vslide corresponds to outx + 2 4 5
          // 6 7 8 9 ( stride == 2) 2nd vslide corresponds to outx + 3 6 7 8 9 (
          // stride == 2)
          FORGING_4_INPUT_COLUMNS_INTO_V0_V7_FOR_ACONV;

          aconv_vxv(v48, v0, cmds, v8);  // filter r1

          // inputs row 1 and row 3
          vld_b_x(v0, p_input +
                          ((in_y_origin + 1) * input_width * input_depth) +
                          (in_x_origin * input_depth));
          vld_b_x(v4, p_input +
                          ((in_y_origin + 3) * input_width * input_depth) +
                          (in_x_origin * input_depth));
          FORGING_4_INPUT_COLUMNS_INTO_V0_V7_FOR_ACONV;
          aconv_vxv(v48, v0, cmds, v24);  // filter r2

          // row 2 and row4
          vld_b_x(v0, p_input +
                          ((in_y_origin + 2) * input_width * input_depth) +
                          (in_x_origin * input_depth));
          vld_b_x(v4, p_input +
                          ((in_y_origin + 4) * input_width * input_depth) +
                          (in_x_origin * input_depth));

          FORGING_4_INPUT_COLUMNS_INTO_V0_V7_FOR_ACONV;
          aconv_vxv(v48, v0, cmds, v40);  // filter r3

          vcget(v48);
          actr_v(v48, v48);
          vcget(v48);

          //     (x0,y0)   (x0, y1)
          // v48 (0 2 1 3  0 2 1 3) -- even registers
          // v49 (4 6 5 7  4 6 5 7) -- odd registers
          POST_PROCESS_ACONV_INT32_ACC_TO_INT8_ACC;

          for (int i = 0; i < 4; ++i) {
            memcpy((p_output + ((out_x + i) * output_depth) +
                    (out_y * output_width * output_depth)),
                   &acc_out8[0][i][0], 8 * sizeof(int8_t));
            memcpy((p_output + ((out_x + i) * output_depth) +
                    ((out_y + 1) * output_width * output_depth)),
                   &acc_out8[1][i][0], 8 * sizeof(int8_t));
          }
        }

        int in_x_origin = (output_width - 4) * stride_width;
        vdup_w_x_m(v48, 0);
        vdup_w_x_m(v52, 0);
        acset_v(v48, v48);

        vld_b_l_xx(v0,
                   p_input + (in_y_origin * input_width * input_depth) +
                       (in_x_origin * input_depth),
                   24);
        vld_b_l_xx(v4,
                   p_input + ((in_y_origin + 2) * input_width * input_depth) +
                       (in_x_origin * input_depth),
                   24);
        vor_vv(v0, v0, v30);
        vor_vv(v4, v4, v30);
        FORGING_4_INPUT_COLUMNS_INTO_V0_V7_FOR_ACONV;
        aconv_vxv(v48, v0, cmds, v8);  // filter r1

        // inputs row 1 and row 3
        vld_b_l_xx(v0,
                   p_input + ((in_y_origin + 1) * input_width * input_depth) +
                       (in_x_origin * input_depth),
                   24);
        vld_b_l_xx(v4,
                   p_input + ((in_y_origin + 3) * input_width * input_depth) +
                       (in_x_origin * input_depth),
                   24);
        vor_vv(v0, v0, v30);
        vor_vv(v4, v4, v30);
        FORGING_4_INPUT_COLUMNS_INTO_V0_V7_FOR_ACONV;
        aconv_vxv(v48, v0, cmds, v24);  // filter r2

        // row 2 and row4
        vld_b_l_xx(v0,
                   p_input + ((in_y_origin + 2) * input_width * input_depth) +
                       (in_x_origin * input_depth),
                   24);
        vld_b_l_xx(v4,
                   p_input + ((in_y_origin + 4) * input_width * input_depth) +
                       (in_x_origin * input_depth),
                   24);
        vor_vv(v0, v0, v30);
        vor_vv(v4, v4, v30);
        FORGING_4_INPUT_COLUMNS_INTO_V0_V7_FOR_ACONV;
        aconv_vxv(v48, v0, cmds, v40);  // filter r3

        vcget(v48);
        actr_v(v48, v48);
        vcget(v48);

        POST_PROCESS_ACONV_INT32_ACC_TO_INT8_ACC;

        for (int i = 0; i < 4; ++i) {
          memcpy((p_output + ((output_width - 4 + i) * output_depth) +
                  (out_y * output_width * output_depth)),
                 &acc_out8[0][i][0], 8 * sizeof(int8_t));

          memcpy((p_output + ((output_width - 4 + i) * output_depth) +
                  ((out_y + 1) * output_width * output_depth)),
                 &acc_out8[1][i][0], 8 * sizeof(int8_t));
        }
      }
      int load_until = 32;
      bool negate_offset = false;
      for (int out_x = 0; out_x + 4 <= output_width; out_x += 4) {
        const int in_x_origin = (out_x * stride_width);
        const int in_y_origin = (output_height - 2) * stride_height;

        if (out_x + 4 == output_width) {
          load_until = 24;
          negate_offset = true;
        }

        vdup_w_x_m(v48, 0);
        vdup_w_x_m(v52, 0);
        acset_v(v48, v48);

        // inputs row 0 and row 2
        vld_b_l_xx(v0,
                   p_input + (in_y_origin * input_width * input_depth) +
                       (in_x_origin * input_depth),
                   load_until);
        vld_b_l_xx(v4,
                   p_input + ((in_y_origin + 2) * input_width * input_depth) +
                       (in_x_origin * input_depth),
                   load_until);
        if (negate_offset) {
          vor_vv(v0, v0, v30);
          vor_vv(v4, v4, v30);
        }
        FORGING_4_INPUT_COLUMNS_INTO_V0_V7_FOR_ACONV
        aconv_vxv(v48, v0, cmds, v8);  // filter r1

        // inputs row 1 and row 3
        vld_b_l_xx(v0,
                   p_input + ((in_y_origin + 1) * input_width * input_depth) +
                       (in_x_origin * input_depth),
                   load_until);
        vld_b_l_xx(v4,
                   p_input + ((in_y_origin + 3) * input_width * input_depth) +
                       (in_x_origin * input_depth),
                   load_until);
        if (negate_offset) {
          vor_vv(v0, v0, v30);
          vor_vv(v4, v4, v30);
        }
        FORGING_4_INPUT_COLUMNS_INTO_V0_V7_FOR_ACONV;
        aconv_vxv(v48, v0, cmds, v24);  // filter r2

        // row 2 and row4
        vld_b_l_xx(v0,
                   p_input + ((in_y_origin + 2) * input_width * input_depth) +
                       (in_x_origin * input_depth),
                   load_until);
        vdup_b_x_m(v4, neg_input_offset);
        if (negate_offset) {
          vor_vv(v0, v0, v30);
          vor_vv(v4, v4, v30);
        }
        vsliden_h_3_vv(v1, v0, v11);
        vsliden_w_3_vv(v2, v0, v11);
        vsliden_w_3_vv(v3, v1, v11);
        aconv_vxv(v48, v0, cmds, v40);  // filter r3

        vcget(v48);
        actr_v(v48, v48);
        vcget(v48);
        POST_PROCESS_ACONV_INT32_ACC_TO_INT8_ACC;

        for (int i = 0; i < 4; ++i) {
          memcpy((p_output + ((out_x + i) * output_depth) +
                  ((output_height - 2) * output_width * output_depth)),
                 &acc_out8[0][i][0], 8 * sizeof(int8_t));

          memcpy((p_output + ((out_x + i) * output_depth) +
                  ((output_height - 1) * output_width * output_depth)),
                 &acc_out8[1][i][0], 8 * sizeof(int8_t));
        }
      }
    }
    out_channel += 8;
  } while (out_channel < output_depth);
}

#undef FORGING_4_INPUT_COLUMNS_INTO_V0_V7_FOR_ACONV
#undef POST_PROCESS_ACONV_INT32_ACC_TO_INT8_ACC

}  // namespace kelvin::opt