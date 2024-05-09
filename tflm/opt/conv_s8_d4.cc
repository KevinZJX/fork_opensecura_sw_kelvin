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
// Special case for filter depth = 4n

#include <cstdlib>
#include <memory>

#include "crt/kelvin.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace kelvin::opt {
namespace {

void Filter_N_H_W_M(const int8_t* input, int8_t* output, int N, int H, int W, int M) {
  const int8_t(&in)[8][H][W][M] = *(int8_t(*)[8][H][W][M])input;
  int8_t(&out)[H][W][M / 4][8][4] = *(int8_t(*)[H][W][M / 4][8][4]) output;
  assert(M >= 4);
  for (int zo = 0; zo < N; ++zo) {
    for (int ky = 0; ky < H; ++ky) {
      for (int kx = 0; kx < W; ++kx) {
        for (int zi = 0; zi < M; ++zi) {
          const int zi_hi = zi >> 2;  // div4
          const int zi_lo = zi & 3;   // rem4
          out[ky][kx][zi_hi][zo][zi_lo] = in[zo][ky][kx][zi];
        }
      }
    }
  }
  // Zero out the rest of the output.
  for (int zo = N; zo < 8; ++zo) {
    for (int ky = 0; ky < H; ++ky) {
      for (int kx = 0; kx < W; ++kx) {
        for (int zi = 0; zi < M; ++zi) {
          const int zi_hi = zi >> 2;  // div4
          const int zi_lo = zi & 3;   // rem4
          out[ky][kx][zi_hi][zo][zi_lo] = 0;
        }
      }
    }
  }
}

void Swizzle(const int32_t* input, int32_t* output, int N) {
  assert(N <= 8);
  const int32_t(&in)[8] = *(int32_t(*)[8])input;
  int32_t(&out)[32] = *(int32_t(*)[32]) output;
  // Convert to accumulator swizzle pattern.
  memset(out, 0, 32 * sizeof(int32_t));
  int offsets[] = {0, 16, 8, 24, 1, 17, 9, 25};
  for (int i = 0; i < N; ++i) {
    int offset = offsets[i];
    out[0 + offset] = out[2 + offset] = out[4 + offset] = out[6 + offset] = in[i];
  }
}

}  // namespace

void ConvS8D4(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int32_t neg_input_offset = -params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
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
  cmds.conv.stop = 7;
  cmds.conv.sbias1 = input_offset;
  cmds.conv.sdata1 = true;
  cmds.conv.sbias2 = 0;
  cmds.conv.sdata2 = true;

  const size_t swizzled_filter_data_size =
      8 * filter_height * filter_width * filter_input_depth;
  std::unique_ptr<int8_t> swizzled_filter_data(reinterpret_cast<int8_t*>(
      ::aligned_alloc(32, swizzled_filter_data_size)));
  int8_t* p_swizzled_filter_data = swizzled_filter_data.get();
  int32_t swizzled_bias_data[32];
  int32_t swizzled_mult_data[32];
  int32_t swizzled_shift_data[32];

  int out_channel = 0;
  do {
    int out_channels_this_iter = std::min(8, output_depth - out_channel);
    Filter_N_H_W_M(filter_data + (out_channel * filter_height * filter_width *
                                  filter_input_depth),
                   p_swizzled_filter_data, out_channels_this_iter, filter_height, filter_width,
                   filter_input_depth);
    Swizzle(bias_data + out_channel, swizzled_bias_data, out_channels_this_iter);
    Swizzle(output_multiplier + out_channel, swizzled_mult_data, out_channels_this_iter);
    Swizzle(output_shift + out_channel, swizzled_shift_data, out_channels_this_iter);
    vld_w_x_m(v16, swizzled_bias_data);
    vld_w_x_m(v20, swizzled_mult_data);
    vld_w_x_m(v24, swizzled_shift_data);
    vrsub_w_vx_m(v24, v24, 0);

    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        int out_x = 0;
        do {
          int out_xs_this_iter = std::min(8, output_width - out_x);
          // 8x accumulators
          vdup_w_x_m(v48, 0);
          vdup_w_x_m(v52, 0);
          acset_v(v48, v48);
          int in_channel = 0;
          do {
            int in_channels_this_iter = std::min(filter_input_depth, 32);
            for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              const bool is_row_inside_input =
                  (in_y >= 0) && (in_y < input_height);
              if (!is_row_inside_input) {
                continue;
              }

              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                int in_x[8];
                bool right_pad = false;
                int first_right_pad = -1;
                for (int i = 0; i < 8; ++i) {
                  const int in_x_origin =
                      ((out_x + i) * stride_width) - pad_width;
                  in_x[i] = in_x_origin + dilation_width_factor * filter_x;
                }
                bool left_pad = (in_x[0] < 0);
                for (int i = 7; i >= 0; --i) {
                  if (in_x[i] < input_width) {
                    break;
                  }
                  right_pad = true;
                  first_right_pad = i;
                }

                if (left_pad) {
                  vdup_b_x(v0, -input_offset);
                  vld_b_s_xx(
                      v1,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[1], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx(
                      v2,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[2], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx(
                      v3,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[3], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx_m(
                      v4,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[4], in_channel)],
                      input_depth * stride_width);
                } else if (right_pad) {
                  int first_pad = std::min(first_right_pad, out_xs_this_iter);
                  switch (first_pad) {
                    case 0:
                      vdup_b_x(v0, neg_input_offset);
                    case 1:
                      vdup_b_x(v1, neg_input_offset);
                    case 2:
                      vdup_b_x(v2, neg_input_offset);
                    case 3:
                      vdup_b_x(v3, neg_input_offset);
                    case 4:
                      vdup_b_x(v4, neg_input_offset);
                    case 5:
                      vdup_b_x(v5, neg_input_offset);
                    case 6:
                      vdup_b_x(v6, neg_input_offset);
                    case 7:
                      vdup_b_x(v7, neg_input_offset);
                  }
                  switch (8 - first_pad) { // rest (stripmines?)
                    case 0:
                      vld_b_s_xx(
                          v7,
                          &input_data[tflite::Offset(input_shape, batch, in_y,
                                                     in_x[7], in_channel)],
                          input_depth * stride_width);
                    case 1:
                      vld_b_s_xx(
                          v6,
                          &input_data[tflite::Offset(input_shape, batch, in_y,
                                                     in_x[6], in_channel)],
                          input_depth * stride_width);
                    case 2:
                      vld_b_s_xx(
                          v5,
                          &input_data[tflite::Offset(input_shape, batch, in_y,
                                                     in_x[5], in_channel)],
                          input_depth * stride_width);
                    case 3:
                      vld_b_s_xx(
                          v4,
                          &input_data[tflite::Offset(input_shape, batch, in_y,
                                                     in_x[4], in_channel)],
                          input_depth * stride_width);
                    case 4:
                      vld_b_s_xx(
                          v3,
                          &input_data[tflite::Offset(input_shape, batch, in_y,
                                                     in_x[3], in_channel)],
                          input_depth * stride_width);
                    case 5:
                      vld_b_s_xx(
                          v2,
                          &input_data[tflite::Offset(input_shape, batch, in_y,
                                                     in_x[2], in_channel)],
                          input_depth * stride_width);
                    case 6:
                      vld_b_s_xx(
                          v1,
                          &input_data[tflite::Offset(input_shape, batch, in_y,
                                                     in_x[1], in_channel)],
                          input_depth * stride_width);
                    case 7:
                      vld_b_s_xx(
                          v0,
                          &input_data[tflite::Offset(input_shape, batch, in_y,
                                                     in_x[0], in_channel)],
                          input_depth * stride_width);
                  }
                } else if (!left_pad && !right_pad) {
                  // Inputs
                  vld_b_s_xx_m(
                      v0,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[0], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx_m(
                      v4,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[4], in_channel)],
                      input_depth * stride_width);
                } else {
                  vdup_b_x(v0, -input_offset);
                  vdup_b_x(v7, -input_offset);
                  vld_b_s_xx_m(
                      v1,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[1], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx(
                      v5,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[5], in_channel)],
                      input_depth * stride_width);
                  vld_b_s_xx(
                      v6,
                      &input_data[tflite::Offset(input_shape, batch, in_y,
                                                 in_x[6], in_channel)],
                      input_depth * stride_width);
                }
                size_t local_filter_offset =
                    (filter_y * filter_width * 8 * input_depth) +
                    (filter_x * 8 * input_depth) + (in_channel * 8);
                int8_t* p_local_filter_start =
                    p_swizzled_filter_data + local_filter_offset;
                vld_b_p_x_m(v8, p_local_filter_start);
                vld_b_x_m(v12, p_local_filter_start);

                cmds.conv.stop = (in_channels_this_iter / 4) - 1;
                aconv_vxv(v48, v0, cmds, v8);
              }
            }
            in_channel += in_channels_this_iter;
          } while (in_channel < filter_input_depth);
          vcget(v48);
          vadd_w_vv_m(v48, v48, v16);
          vadd_w_vv_m(v52, v52, v16);
          vdmulh_w_rn_vv_m(v48, v48, v20);
          vdmulh_w_rn_vv_m(v52, v52, v20);
          vsha_w_r_vv_m(v48, v48, v24);
          vsha_w_r_vv_m(v52, v52, v24);
          vadd_w_vx_m(v48, v48, output_offset);
          vadd_w_vx_m(v52, v52, output_offset);
          vmin_w_vx_m(v48, v48, output_activation_max);
          vmin_w_vx_m(v52, v52, output_activation_max);
          vmax_w_vx_m(v48, v48, output_activation_min);
          vmax_w_vx_m(v52, v52, output_activation_min);
          vsraqs_b_vx(v56, v48, 0);
          vsraqs_b_vx(v57, v52, 0);
          if (out_channels_this_iter == 8) {
            if (out_xs_this_iter >= 4) {
              vstq_b_s_xx(v56,
                          &output_data[tflite::Offset(output_shape, batch, out_y,
                                                      out_x, out_channel)],
                          output_depth);
            } else {
              for (int i = 0; i < std::min(4, out_xs_this_iter); ++i) {
                if (i > 0) {
                  vsliden_b_4_vv(v58, v56, v0);
                  vsliden_b_4_vv(v56, v58, v0);
                }
                vst_b_l_xx(v56,
                          &output_data[tflite::Offset(output_shape, batch, out_y,
                                                      out_x + i, out_channel)],
                          out_channels_this_iter);
              }
            }
            if (out_xs_this_iter == 8) {
              vstq_b_s_xx(v57,
                          &output_data[tflite::Offset(output_shape, batch, out_y,
                                                      out_x + 4, out_channel)],
                          output_depth);
            } else if (out_xs_this_iter > 4) {
              for (int i = 4; i < std::min(8, out_xs_this_iter); ++i) {
                if (i > 4) {
                  vsliden_b_4_vv(v58, v57, v0);
                  vsliden_b_4_vv(v57, v58, v0);
                }
                vst_b_l_xx(v57,
                          &output_data[tflite::Offset(output_shape, batch, out_y,
                                                      out_x + i, out_channel)],
                          out_channels_this_iter);
              }
            }
          } else {
              for (int i = 0; i < std::min(4, out_xs_this_iter); ++i) {
                if (i > 0) {
                  vsliden_b_4_vv(v58, v56, v0);
                  vsliden_b_4_vv(v56, v58, v0);
                }
                vst_b_l_xx(v56,
                          &output_data[tflite::Offset(output_shape, batch, out_y,
                                                      out_x + i, out_channel)],
                          out_channels_this_iter);
              }
            if (out_xs_this_iter > 4) {
              for (int i = 4; i < std::min(8, out_xs_this_iter); ++i) {
                if (i > 4) {
                  vsliden_b_4_vv(v58, v57, v0);
                  vsliden_b_4_vv(v57, v58, v0);
                }
                vst_b_l_xx(v57,
                          &output_data[tflite::Offset(output_shape, batch, out_y,
                                                      out_x + i, out_channel)],
                          out_channels_this_iter);
              }
            }
          }
          out_x += out_xs_this_iter;
        } while (out_x < output_width);
      }
    }
    out_channel += out_channels_this_iter;
  } while (out_channel < output_depth);
}
}  // namespace kelvin::opt
