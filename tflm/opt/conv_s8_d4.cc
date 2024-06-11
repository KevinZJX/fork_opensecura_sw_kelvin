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
#include "tflm/opt/conv_s8.h"
#include "tflm/opt/conv_util.h"

#define unlikely(x) (__builtin_expect(false || (x), false))
#define likely(x) (__builtin_expect(false || (x), true))

namespace kelvin::opt {
namespace {

// Version of Filter_N_H_W_M which also pads outputs to 8 and inputs to 4.
void PaddedFilter_N_H_W_M(const int8_t* input, int8_t* output, int N, int H,
                          int W, int M) {
  const int8_t(&in)[8][H][W][M] = *(int8_t(*)[8][H][W][M])input;
  int8_t(&out)[H][W][M / 4][8][4] = *(int8_t(*)[H][W][M / 4][8][4]) output;
  // assert(M >= 4);
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
        for (int zi = M; zi < 4; ++zi) {
          const int zi_hi = zi >> 2;  // div4
          const int zi_lo = zi & 3;   // rem4
          out[ky][kx][zi_hi][zo][zi_lo] = 0;
        }
      }
    }
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
    PaddedFilter_N_H_W_M(
        filter_data + (out_channel * filter_height * filter_width *
                       filter_input_depth),
        p_swizzled_filter_data, out_channels_this_iter, filter_height,
        filter_width, filter_input_depth);
    Swizzle(bias_data + out_channel, swizzled_bias_data, out_channels_this_iter);
    Swizzle(output_multiplier + out_channel, swizzled_mult_data, out_channels_this_iter);
    Swizzle(output_shift + out_channel, swizzled_shift_data, out_channels_this_iter);
    vld_w_x_m(v16, swizzled_bias_data);
    vld_w_x_m(v20, swizzled_mult_data);
    vld_w_x_m(v24, swizzled_shift_data);
    vrsub_w_vx_m(v24, v24, 0);

    for (int batch = 0; batch < batches; ++batch) {
      const int8_t* p_output =
          output_data + (batch * output_height * output_width * output_depth) +
          out_channel;
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        const int out_y_offset = (out_y * output_width * output_depth);
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
            // Calculate first valid filter_y
            int filter_y = 0;
            {
              int in_y = in_y_origin;
              while (in_y < 0) {
                ++filter_y;
                in_y += (dilation_height_factor);
              }
            }
            for (; filter_y < filter_height; ++filter_y) {
              const int y_filter_offset =
                  (filter_y * filter_width * 8 * input_depth);
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              if (in_y >= input_height) {
                break;
              }
              const int8_t* p_in =
                  input_data + in_channel + (in_y * input_width * input_depth) +
                  (batch * input_height * input_width * input_depth);

              int in_x[8];
#pragma GCC unroll 8
              for (int i = 0; i < 8; ++i) {
                in_x[i] = ((out_x + i) * stride_width) - pad_width;
              }
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int8_t* p_in_x[8];
                int first_right_pad = -1;

#pragma GCC unroll 8
                for (int i = 0; i < 8; ++i) {
                  p_in_x[i] = p_in + (in_x[i] * input_depth);
                }

#pragma GCC unroll 8
                for (int i = 7; i >= 0; --i) {
                  if (in_x[i] < input_width) {
                    break;
                  }
                  first_right_pad = i;
                }
                bool left_pad = (in_x[0] < 0);
                bool right_pad = (first_right_pad != -1);

                int stride = input_depth * stride_width;

                if (unlikely(left_pad)) {
                  vdup_b_x(v0, -input_offset);
                  vld_b_s_xx(v1, p_in_x[1], stride);
                  vld_b_s_xx(v2, p_in_x[2], stride);
                  vld_b_s_xx(v3, p_in_x[3], stride);
                  vld_b_s_xx_m(v4, p_in_x[4], stride);
                } else if (unlikely(right_pad)) {
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
                      vld_b_s_xx(v7, p_in_x[7], stride);
                    case 1:
                      vld_b_s_xx(v6, p_in_x[6], stride);
                    case 2:
                      vld_b_s_xx(v5, p_in_x[5], stride);
                    case 3:
                      vld_b_s_xx(v4, p_in_x[4], stride);
                    case 4:
                      vld_b_s_xx(v3, p_in_x[3], stride);
                    case 5:
                      vld_b_s_xx(v2, p_in_x[2], stride);
                    case 6:
                      vld_b_s_xx(v1, p_in_x[1], stride);
                    case 7:
                      vld_b_s_xx(v0, p_in_x[0], stride);
                  }
                } else if (likely(!left_pad && !right_pad)) {
                  // Inputs
                  vld_b_s_xx_m(v0, p_in_x[0], stride);
                  vld_b_s_xx_m(v4, p_in_x[4], stride);
                } else {
                  vdup_b_x(v0, neg_input_offset);
                  vdup_b_x(v7, neg_input_offset);
                  vld_b_s_xx_m(v1, p_in_x[1], stride);
                  vld_b_s_xx(v5, p_in_x[5], stride);
                  vld_b_s_xx(v6, p_in_x[6], stride);
                }
                size_t local_filter_offset = y_filter_offset +
                                             (filter_x * 8 * input_depth) +
                                             (in_channel * 8);
                int8_t* p_local_filter_start =
                    p_swizzled_filter_data + local_filter_offset;
                vld_b_p_x_m(v8, p_local_filter_start);
                vld_b_x_m(v12, p_local_filter_start);

                cmds.conv.stop = (in_channels_this_iter / 4) - 1;
                aconv_vxv(v48, v0, cmds, v8);

#pragma GCC unroll 8
                for (int i = 0; i < 8; ++i) {
                  in_x[i] += dilation_width_factor;
                }
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

          const int8_t* p_out_x[8];
#pragma GCC unroll 8
          for (int i = 0; i < 8; ++i) {
            p_out_x[i] = p_output + out_y_offset + ((out_x + i) * output_depth);
          }

          vslidep_h_4_vv(v58, v57, v57);  // x7
          vslidep_h_4_vv(v59, v58, v58);  // x6
          vslidep_h_4_vv(v60, v59, v59);  // x5
          vslidep_h_4_vv(v61, v60, v60);  // x4
          vslidep_h_4_vv(v62, v56, v56);  // x3
          vslidep_h_4_vv(v63, v62, v62);  // x2
          vslidep_h_4_vv(v57, v63, v63);  // x1
          vslidep_h_4_vv(v56, v57, v57);  // x0
          switch (out_xs_this_iter) {
            case 8:
              vst_b_l_xx(v58, p_out_x[7], out_channels_this_iter);
            case 7:
              vst_b_l_xx(v59, p_out_x[6], out_channels_this_iter);
            case 6:
              vst_b_l_xx(v60, p_out_x[5], out_channels_this_iter);
            case 5:
              vst_b_l_xx(v61, p_out_x[4], out_channels_this_iter);
            case 4:
              vst_b_l_xx(v62, p_out_x[3], out_channels_this_iter);
            case 3:
              vst_b_l_xx(v63, p_out_x[2], out_channels_this_iter);
            case 2:
              vst_b_l_xx(v57, p_out_x[1], out_channels_this_iter);
            case 1:
              vst_b_l_xx(v56, p_out_x[0], out_channels_this_iter);
          }
          out_x += out_xs_this_iter;
        } while (out_x < output_width);
      }
    }
    out_channel += out_channels_this_iter;
  } while (out_channel < output_depth);
}

// Optimized for width >= 8
void ConvS8W8D4(
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
    PaddedFilter_N_H_W_M(
        filter_data + (out_channel * filter_height * filter_width *
                       filter_input_depth),
        p_swizzled_filter_data, out_channels_this_iter, filter_height,
        filter_width, filter_input_depth);

    if (bias_data) {
      Swizzle(bias_data + out_channel, swizzled_bias_data, 8);
      vld_w_x_m(v44, swizzled_bias_data);
    } else {
      vdup_w_x_m(v44, 0);
    }

    Swizzle(output_multiplier + out_channel, swizzled_mult_data, out_channels_this_iter);
    Swizzle(output_shift + out_channel, swizzled_shift_data, out_channels_this_iter);

    vld_w_x_m(v56, swizzled_mult_data);
    vld_w_x_m(v60, swizzled_shift_data);
    vrsub_w_vx_m(v60, v60, 0);

    for (int batch = 0; batch < batches; ++batch) {
      int8_t* p_output =
          output_data + (batch * output_height * output_width * output_depth) +
          out_channel;
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        const int out_y_offset = (out_y * output_width * output_depth);
        int out_x = 0;
        while ((out_x * stride_width) < pad_width) {
          int out_xs_this_iter = 8;
          // 8x accumulators
          vmv_v_m(v48, v44);
          vmv_v_m(v52, v44);
          acset_v(v48, v48);

          int in_channel = 0;
          while (in_channel < filter_input_depth) {
            int in_channels_this_iter = std::min(filter_input_depth, 32);
            // Calculate first valid filter_y
            int filter_y = 0;
            {
              int in_y = in_y_origin;
              while (in_y < 0) {
                ++filter_y;
                in_y += (dilation_height_factor);
              }
            }
            for (; filter_y < filter_height; ++filter_y) {
              const int y_filter_offset =
                  (filter_y * filter_width * 8 * input_depth);
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              if (in_y >= input_height) {
                break;
              }
              const int8_t* p_in =
                  input_data + in_channel + (in_y * input_width * input_depth) +
                  (batch * input_height * input_width * input_depth);

              int in_x[8];
#pragma GCC unroll 8
              for (int i = 0; i < 8; ++i) {
                in_x[i] = ((out_x + i) * stride_width) - pad_width;
              }

              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int8_t* p_in_x[8];

#pragma GCC unroll 8
                for (int i = 0; i < 8; ++i) {
                  p_in_x[i] = p_in + (in_x[i] * input_depth);
                }

                int stride = input_depth * stride_width;

                if (in_x[0] < 0) {
                  vdup_b_x(v0, -input_offset);
                  vld_b_s_xx(v1, p_in_x[1], stride);
                  vld_b_s_xx(v2, p_in_x[2], stride);
                  vld_b_s_xx(v3, p_in_x[3], stride);
                  vld_b_s_xx_m(v4, p_in_x[4], stride);
                } else {
                  // Inputs
                  vld_b_s_xx_m(v0, p_in_x[0], stride);
                  vld_b_s_xx_m(v4, p_in_x[4], stride);
                }
                size_t local_filter_offset = y_filter_offset +
                                             (filter_x * 8 * input_depth) +
                                             (in_channel * 8);
                int8_t* p_local_filter_start =
                    p_swizzled_filter_data + local_filter_offset;
                vld_b_p_x_m(v8, p_local_filter_start);
                vld_b_x_m(v12, p_local_filter_start);

                cmds.conv.stop = (in_channels_this_iter / 4) - 1;
                aconv_vxv(v48, v0, cmds, v8);

#pragma GCC unroll 8
                for (int i = 0; i < 8; ++i) {
                  in_x[i] += dilation_width_factor;
                }
              }
            }
            in_channel += in_channels_this_iter;
          }

          vcget(v48);
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
              v48, v52, v56, v60, output_activation_min, output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vsraqs_b_vx(v52, v52, 0);
          int i = 0;
          int8_t* p_out = p_output + out_y_offset + (out_x * output_depth);
          for (; i < std::min(4, out_xs_this_iter); i++) {
            vst_b_l_xx(v48, p_out, out_channels_this_iter);
            p_out += output_depth;
            vsliden_h_4_vv(v48, v48, v48);
          }
          for (; i < out_xs_this_iter; i++) {
            vst_b_l_xx(v52, p_out, out_channels_this_iter);
            p_out += output_depth;
            vsliden_h_4_vv(v52, v52, v52);
          }

          out_x += out_xs_this_iter;
        }  // ((out_x * stride_width) < pad_width)

        // Hot loop, no x padding
        int right_x = ((out_x + 7) * stride_width) + filter_width - pad_width;
        while (right_x < output_width) {
          int out_xs_this_iter = 8;
          // 8x accumulators
          vmv_v_m(v48, v44);
          vmv_v_m(v52, v44);
          acset_v(v48, v48);
          int in_channel = 0;
          while (in_channel < filter_input_depth) {
            int in_channels_this_iter = std::min(filter_input_depth, 32);
            cmds.conv.stop = (in_channels_this_iter / 4) - 1;

            // Calculate first valid filter_y
            int filter_y = 0;
            {
              int in_y = in_y_origin;
              while (in_y < 0) {
                ++filter_y;
                in_y += (dilation_height_factor);
              }
            }
            for (; filter_y < filter_height; ++filter_y) {
              const int y_filter_offset =
                  (filter_y * filter_width * 8 * input_depth);
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              if (in_y >= input_height) {
                break;
              }
              const int8_t* p_in =
                  input_data + in_channel + (in_y * input_width * input_depth) +
                  (batch * input_height * input_width * input_depth);

              int in_x = (out_x * stride_width) - pad_width;

              for (int s = 0; s < stride_width; s++) {
                int filter_x = s;
                int stride = input_depth * stride_width;

                const int8_t* p_in_x0 = p_in +
                    ((in_x + filter_x) * input_depth);
                vld_b_s_xx_m(v0, p_in_x0, stride);
                p_in_x0 += 4 * stride;
                vld_b_s_xx_m(v4, p_in_x0, stride);
                p_in_x0 += 4 * stride;

                {
                  size_t local_filter_offset = y_filter_offset +
                                              (filter_x * 8 * input_depth) +
                                              (in_channel * 8);
                  int8_t* p_local_filter_start =
                      p_swizzled_filter_data + local_filter_offset;
                  vld_b_p_x_m(v8, p_local_filter_start);
                  vld_b_x_m(v12, p_local_filter_start);
                }

                aconv_vxv(v48, v0, cmds, v8);
                filter_x += stride_width;

                for (; filter_x + stride_width < filter_width;
                       filter_x += 2 * stride_width) {
                  // Iteration 1
                  vmv_v(v16, v1);
                  vmv_v(v17, v2);
                  vmv_v(v18, v3);
                  vmv_v(v19, v4);
                  vmv_v(v20, v5);
                  vmv_v(v21, v6);
                  vmv_v(v22, v7);
                  vld_b_l_xx(v23, p_in_x0, in_channels_this_iter);
                  p_in_x0 += stride;

                  size_t local_filter_offset0 = y_filter_offset +
                              (filter_x * 8 * input_depth) +
                              (in_channel * 8);
                  int8_t* p_local_filter_start0 =
                      p_swizzled_filter_data + local_filter_offset0;
                  vld_b_x_m(v24, p_local_filter_start0);
                  vld_b_x_m(v28, p_local_filter_start0 + 128);

                  aconv_vxv(v48, v16, cmds, v24);

                  // Iteration 2
                  vmv_v(v0, v17);
                  vmv_v(v1, v18);
                  vmv_v(v2, v19);
                  vmv_v(v3, v20);
                  vmv_v(v4, v21);
                  vmv_v(v5, v22);
                  vmv_v(v6, v23);
                  vld_b_l_xx(v7, p_in_x0, in_channels_this_iter);
                  p_in_x0 += stride;

                  size_t local_filter_offset1 = y_filter_offset +
                              ((filter_x + stride_width) * 8 * input_depth) +
                              (in_channel * 8);
                  int8_t* p_local_filter_start1 =
                      p_swizzled_filter_data + local_filter_offset1;
                  vld_b_x_m(v8, p_local_filter_start1);
                  vld_b_x_m(v12, p_local_filter_start1 + 128);

                  aconv_vxv(v48, v0, cmds, v8);
                }

                for (; filter_x < filter_width; filter_x += stride_width) {
                  // Iteration 1
                  vmv_v(v16, v1);
                  vmv_v(v17, v2);
                  vmv_v(v18, v3);
                  vmv_v(v19, v4);
                  vmv_v(v20, v5);
                  vmv_v(v21, v6);
                  vmv_v(v22, v7);
                  vld_b_l_xx(v23, p_in_x0, in_channels_this_iter);
                  p_in_x0 += stride;

                  size_t local_filter_offset = y_filter_offset +
                              (filter_x * 8 * input_depth) +
                              (in_channel * 8);
                  int8_t* p_local_filter_start =
                      p_swizzled_filter_data + local_filter_offset;
                  vld_b_x_m(v24, p_local_filter_start);
                  vld_b_x_m(v28, p_local_filter_start + 128);

                  aconv_vxv(v48, v16, cmds, v24);
                }
              }
            }
            in_channel += in_channels_this_iter;
          }  // while (in_channel < filter_input_depth);
          vcget(v48);
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
              v48, v52, v56, v60, output_activation_min, output_activation_max,
              output_offset);

          vsraqs_b_vx(v48, v48, 0);
          vsraqs_b_vx(v52, v52, 0);
          int i = 0;
          int8_t* p_out = p_output + out_y_offset + (out_x * output_depth);
          for (; i < std::min(4, out_xs_this_iter); i++) {
            vst_b_l_xx(v48, p_out, out_channels_this_iter);
            p_out += output_depth;
            vsliden_h_4_vv(v48, v48, v48);
          }
          for (; i < out_xs_this_iter; i++) {
            vst_b_l_xx(v52, p_out, out_channels_this_iter);
            p_out += output_depth;
            vsliden_h_4_vv(v52, v52, v52);
          }

          right_x += out_xs_this_iter * stride_width;
          out_x += out_xs_this_iter;
        }

        while (out_x < output_width) {
          int out_xs_this_iter = std::min(8, output_width - out_x);
          // 8x accumulators
          vmv_v_m(v48, v44);
          vmv_v_m(v52, v44);
          acset_v(v48, v48);
          int in_channel = 0;

          while (in_channel < filter_input_depth) {
            int in_channels_this_iter = std::min(filter_input_depth, 32);
            // Calculate first valid filter_y
            int filter_y = 0;
            {
              int in_y = in_y_origin;
              while (in_y < 0) {
                ++filter_y;
                in_y += (dilation_height_factor);
              }
            }
            for (; filter_y < filter_height; ++filter_y) {
              const int y_filter_offset =
                  (filter_y * filter_width * 8 * input_depth);
              const int in_y = in_y_origin + dilation_height_factor * filter_y;
              if (in_y >= input_height) {
                break;
              }
              const int8_t* p_in =
                  input_data + in_channel + (in_y * input_width * input_depth) +
                  (batch * input_height * input_width * input_depth);

              int in_x[8];
#pragma GCC unroll 8
              for (int i = 0; i < 8; ++i) {
                in_x[i] = ((out_x + i) * stride_width) - pad_width;
              }
              for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
                const int8_t* p_in_x[8];
                int first_right_pad = -1;

#pragma GCC unroll 8
                for (int i = 0; i < 8; ++i) {
                  p_in_x[i] = p_in + (in_x[i] * input_depth);
                }

#pragma GCC unroll 8
                for (int i = 7; i >= 0; --i) {
                  if (in_x[i] < input_width) {
                    break;
                  }
                  first_right_pad = i;
                }
                bool left_pad = (in_x[0] < 0);
                bool right_pad = (first_right_pad != -1);

                int stride = input_depth * stride_width;

                if (unlikely(left_pad)) {
                  vdup_b_x(v0, -input_offset);
                  vld_b_s_xx(v1, p_in_x[1], stride);
                  vld_b_s_xx(v2, p_in_x[2], stride);
                  vld_b_s_xx(v3, p_in_x[3], stride);
                  vld_b_s_xx_m(v4, p_in_x[4], stride);
                } else if (unlikely(right_pad)) {
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
                      vld_b_s_xx(v7, p_in_x[7], stride);
                    case 1:
                      vld_b_s_xx(v6, p_in_x[6], stride);
                    case 2:
                      vld_b_s_xx(v5, p_in_x[5], stride);
                    case 3:
                      vld_b_s_xx(v4, p_in_x[4], stride);
                    case 4:
                      vld_b_s_xx(v3, p_in_x[3], stride);
                    case 5:
                      vld_b_s_xx(v2, p_in_x[2], stride);
                    case 6:
                      vld_b_s_xx(v1, p_in_x[1], stride);
                    case 7:
                      vld_b_s_xx(v0, p_in_x[0], stride);
                  }
                } else if (likely(!left_pad && !right_pad)) {
                  // Inputs
                  vld_b_s_xx_m(v0, p_in_x[0], stride);
                  vld_b_s_xx_m(v4, p_in_x[4], stride);
                } else {
                  vdup_b_x(v0, neg_input_offset);
                  vdup_b_x(v7, neg_input_offset);
                  vld_b_s_xx_m(v1, p_in_x[1], stride);
                  vld_b_s_xx(v5, p_in_x[5], stride);
                  vld_b_s_xx(v6, p_in_x[6], stride);
                }
                size_t local_filter_offset = y_filter_offset +
                                             (filter_x * 8 * input_depth) +
                                             (in_channel * 8);
                int8_t* p_local_filter_start =
                    p_swizzled_filter_data + local_filter_offset;
                vld_b_p_x_m(v8, p_local_filter_start);
                vld_b_x_m(v12, p_local_filter_start);

                cmds.conv.stop = (in_channels_this_iter / 4) - 1;
                aconv_vxv(v48, v0, cmds, v8);

#pragma GCC unroll 8
                for (int i = 0; i < 8; ++i) {
                  in_x[i] += dilation_width_factor;
                }
              }
            }
            in_channel += in_channels_this_iter;
          }  // while (in_channel < filter_input_depth);

          vcget(v48);
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
              v48, v52, v56, v60, output_activation_min, output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vsraqs_b_vx(v52, v52, 0);

          int i = 0;
          int8_t* p_out = p_output + out_y_offset + (out_x * output_depth);
          for (; i < std::min(4, out_xs_this_iter); i++) {
            vst_b_l_xx(v48, p_out, out_channels_this_iter);
            p_out += output_depth;
            vsliden_h_4_vv(v48, v48, v48);
          }
          for (; i < out_xs_this_iter; i++) {
            vst_b_l_xx(v52, p_out, out_channels_this_iter);
            p_out += output_depth;
            vsliden_h_4_vv(v52, v52, v52);
          }

          out_x += out_xs_this_iter;
        } // while (out_x < output_width);
      }
    }
    out_channel += out_channels_this_iter;
  } while (out_channel < output_depth);
}

}  // namespace kelvin::opt
