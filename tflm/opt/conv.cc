/*
 * Copyright 2023 Google LLC
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

#include <cassert>
#include <memory>

#include "crt/kelvin.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tflm/opt/opt.h"
#include "tflm/opt/util.h"

namespace kelvin::opt {
namespace {
/* clang-format off */
constexpr const int swizzle[16] = {
    0, 4, 8, 12,
    2, 6, 10, 14,
    1, 5, 9, 13,
    3, 7, 11, 15,
};
/* clang-format on */

constexpr int kFilterHeightIndex = 1;
constexpr int kFilterWidthIndex = 2;
constexpr int kFilterInputChannelIndex = 3;
constexpr int kInputChannelIndex = 3;
constexpr int kOutputChannelIndex = 3;
}  // namespace

void conv_per_channel_b32(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data) {
  const auto batches = MatchingDim(input_shape, 0, output_shape, 0);
  const auto stride_width = params.stride_width;
  const auto stride_height = params.stride_height;
  const auto dilation_width_factor = params.dilation_width_factor;
  const auto dilation_height_factor = params.dilation_height_factor;
  const auto pad_width = params.padding_values.width;
  const auto pad_height = params.padding_values.height;
  const auto input_height = input_shape.Dims(1);
  const auto input_width = input_shape.Dims(2);
  const auto input_depth = input_shape.Dims(3);
  const auto input_offset = params.input_offset;
  const auto filter_height = filter_shape.Dims(1);
  const auto filter_width = filter_shape.Dims(2);
  const auto filter_depth = filter_shape.Dims(3);
  const auto output_height = output_shape.Dims(1);
  const auto output_width = output_shape.Dims(2);
  const auto output_depth = output_shape.Dims(3);
  const auto output_offset = params.output_offset;
  const auto output_activation_min = params.quantized_activation_min;
  const auto output_activation_max = params.quantized_activation_max;
  const auto groups = input_depth / filter_depth;
  const auto filters_per_group = output_depth / groups;

  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = out_y * stride_height - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = out_x * stride_width - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          int32_t acc32 = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              const bool inside = (in_x >= 0) && (in_x < input_width) &&
                                  (in_y >= 0) && (in_y < input_height);
              if (!inside) {
                continue;
              }
              int in_channel = 0;
              do {
                int load_count = std::min(filter_depth - in_channel, 16L);
                int32_t input_swizzled[16];
                const int16_t* p_input = &input_data[tflite::Offset(
                    input_shape, batch, in_y, in_x,
                    in_channel + group * filter_depth)];
                for (int i = 0; i < 16; ++i) {
                  int swizzle_idx = swizzle[i];
                  if (swizzle_idx < load_count)
                    input_swizzled[i] = *(p_input + swizzle_idx) + input_offset;
                  else
                    input_swizzled[i] = 0;
                }
                vld_w_l_xx(v0, input_swizzled, 4);
                vld_w_l_xx(v1, input_swizzled + 4, 4);
                vld_w_l_xx(v2, input_swizzled + 8, 4);
                vld_w_l_xx(v3, input_swizzled + 12, 4);
                vld_b_l_xx(v4,
                           &filter_data[tflite::Offset(filter_shape,
                                                       out_channel, filter_y,
                                                       filter_x, in_channel)],
                           load_count);
                vaddw_h_vx(v4, v4, 0);
                vaddw_w_vx(v6, v5, 0);
                vaddw_w_vx(v4, v4, 0);

                vmul_w_vv_m(vm0, vm0, vm1);
                vadd_w_vv(v0, v0, v1);
                vadd_w_vv(v0, v0, v2);
                vadd_w_vv(v0, v0, v3);
                int32_t acc_spill[4];
                vst_w_l_xx(v0, acc_spill, 4);
                for (int i = 0; i < 4; ++i) {
                  acc32 += acc_spill[i];
                }
                in_channel += 16;
              } while (in_channel + 16 <= filter_depth);
            }
          }
          if (bias_data) {
            acc32 = acc32 + bias_data[out_channel];
          }
          int32_t acc = tflite::MultiplyByQuantizedMultiplier(
              acc32, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::clamp(acc, output_activation_min, output_activation_max);
          output_data[tflite::Offset(output_shape, batch, out_y, out_x,
                                     out_channel)] = static_cast<int16_t>(acc);
        }
      }
    }
  }
}

// Accumulates in v0-v7. [v0-v3], [v4-v7] are sub accumulators for two outputs.
// Load/swizzle filters use [v52-v63].
// Input activations use [v32-v33].
// No clobbers.
void ukernel_s8_s16(const int16_t* input_data0,
                    const int8_t* filter_data0,
                    const int8_t* filter_data1,
                    size_t n) {
  n = n >> 5;
  while (n > 0) {
    // Load filters 0 to v58, v59
    vld_b_p_x(v52, filter_data0);
    vaddw_h_vx(v56, v52, 0);
    vzip_h_vv(v58, v56, v57);

    // Load activations
    vld_h_p_x(v32, input_data0);
    vld_h_p_x(v33, input_data0);

    // Multiply filters0 * activations
    vmulw_w_vv(v16, v58, v32);
    vmulw_w_vv(v18, v59, v33);

    // Accumulate v0
    vadd_w_vv_m(v0, v0, v16);

    // Load filters 1 to v62, v63
    vld_b_p_x(v53, filter_data1);
    vaddw_h_vx(v60, v53, 0);
    vzip_h_vv(v62, v60, v61);

    // Multiply filters1 * activations
    vmulw_w_vv(v20, v62, v32);
    vmulw_w_vv(v22, v63, v33);

    // Accumulate v4
    vadd_w_vv_m(v4, v4, v20);
    n--;
  }
}

void conv_per_channel_b64_1x1(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data) {
  const auto batches = MatchingDim(input_shape, 0, output_shape, 0);
  const auto input_height = input_shape.Dims(1);
  const auto input_width = input_shape.Dims(2);
  const auto input_depth = input_shape.Dims(3);
  const auto input_offset = params.input_offset;
  const auto filter_input_depth = filter_shape.Dims(3);
  const auto output_depth = output_shape.Dims(3);
  const auto output_offset = params.output_offset;
  const auto output_activation_min = params.quantized_activation_min;
  const auto output_activation_max = params.quantized_activation_max;
  const auto groups = input_depth / filter_input_depth;
  const auto output_filters_per_group = output_depth / groups;

  int32_t accumulators[8];
  for (int bhw = 0; bhw < batches * input_height * input_width; bhw++) {
    const int16_t* local_input = input_data + (bhw * input_depth);
    int16_t* local_output = output_data + (bhw * output_depth);
    for (int g = 0; g < groups; g++) {
      const int16_t* group_input = local_input + (g * filter_input_depth);
      for (int gc = 0; gc + 2 <= output_filters_per_group; gc += 2) {
        int oc = (g * output_filters_per_group) + gc;
        const int8_t* local_filters0 = filter_data + (oc * filter_input_depth);
        const int8_t* local_filters1 = local_filters0 + filter_input_depth;

        vdup_w_x_m(v0, 0);
        vdup_w_x_m(v4, 0);
        ukernel_s8_s16(group_input, local_filters0, local_filters1,
                       filter_input_depth);
        // sum accumulators
        vadd_w_vv(v0, v0, v1);
        vadd_w_vv(v2, v2, v3);
        vadd_w_vv(v0, v0, v2);
        vadd_w_vv(v4, v4, v5);
        vadd_w_vv(v6, v6, v7);
        vadd_w_vv(v4, v4, v6);

        {
          vst_w_x(v0, accumulators);
          int64_t acc64 = bias_data[oc];
          for (int i = 0; i < 8; i++) {
            acc64 += accumulators[i];
          }
          int32_t acc = tflite::MultiplyByQuantizedMultiplier(
                acc64, output_multiplier[oc], output_shift[oc]);
          acc += output_offset;
          acc = std::clamp(acc, output_activation_min, output_activation_max);
          local_output[oc] = static_cast<int16_t>(acc);
        }

        {
          vst_w_x(v4, accumulators);
          int64_t acc64 = bias_data[oc + 1];
          for (int i = 0; i < 8; i++) {
            acc64 += accumulators[i];
          }
          int32_t acc = tflite::MultiplyByQuantizedMultiplier(
                acc64, output_multiplier[oc + 1], output_shift[oc + 1]);
          acc += output_offset;
          acc = std::clamp(acc, output_activation_min, output_activation_max);
          local_output[oc + 1] = static_cast<int16_t>(acc);
        }
      }
    }
  }
}

// Optimized for grouped convolutions, no dilation, 1xn filter
void conv_per_channel_b64_filter1xn_group(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data) {
  const auto batches = MatchingDim(input_shape, 0, output_shape, 0);
  const auto stride_width = params.stride_width;
  const auto pad_width = params.padding_values.width;
  const auto input_width = input_shape.Dims(2);
  const auto input_depth = input_shape.Dims(3);
  const auto input_offset = params.input_offset;
  const auto filter_width = filter_shape.Dims(2);
  const auto filter_depth = filter_shape.Dims(3);
  const auto output_width = output_shape.Dims(2);
  const auto output_depth = output_shape.Dims(3);
  const auto output_offset = params.output_offset;
  const auto output_activation_min = params.quantized_activation_min;
  const auto output_activation_max = params.quantized_activation_max;

  const auto groups = input_depth / filter_depth;
  const auto output_filters_per_group = output_depth / groups;

  int32_t accumulators[8];
  for (int g = 0; g < groups; g++) {
    for (int gc = 0; gc + 2 <= output_filters_per_group; gc += 2) {
      int oc = (g * output_filters_per_group) + gc;
      for (int b = 0; b < batches; ++b) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = out_x * stride_width - pad_width;
          const int8_t* local_filters0 =
              filter_data + (oc * filter_width * filter_depth);
          const int8_t* local_filters1 =
              local_filters0 + (filter_width * filter_depth);
          const int16_t* local_input = input_data +
            (b * input_width * input_depth) +
            (in_x_origin * input_depth) +
            (g * filter_depth);
          int16_t* local_output = output_data +
              (b * output_width * output_depth) +
              (out_x * output_depth);

          int64_t acc64_0 = 0;
          int64_t acc64_1 = 0;
          vdup_w_x_m(v0, 0);
          vdup_w_x_m(v4, 0);
          for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
            const int8_t* local_filters0x =
                local_filters0 + (filter_x * filter_depth);
            const int8_t* local_filters1x =
                local_filters1 + (filter_x * filter_depth);
            const int16_t* local_inputx =
                local_input + (filter_x * input_depth);

            ukernel_s8_s16(local_inputx, local_filters0x, local_filters1x,
                           filter_depth);
          }

          // sum accumulators
          vadd_w_vv(v0, v0, v1);
          vadd_w_vv(v2, v2, v3);
          vadd_w_vv(v0, v0, v2);
          vadd_w_vv(v4, v4, v5);
          vadd_w_vv(v6, v6, v7);
          vadd_w_vv(v4, v4, v6);

          {
            vst_w_x(v0, accumulators);
            for (int i = 0; i < 8; i++) {
              acc64_0 += accumulators[i];
            }
            acc64_0 += bias_data[oc];
            int32_t acc = tflite::MultiplyByQuantizedMultiplier(
                  acc64_0, output_multiplier[oc], output_shift[oc]);
            acc += output_offset;
            acc = std::clamp(acc, output_activation_min, output_activation_max);
            local_output[oc] = static_cast<int16_t>(acc);
          }

          {
            vst_w_x(v4, accumulators);
            for (int i = 0; i < 8; i++) {
              acc64_1 += accumulators[i];
            }
            acc64_1 += bias_data[oc + 1];
            int32_t acc = tflite::MultiplyByQuantizedMultiplier(
                  acc64_1, output_multiplier[oc + 1], output_shift[oc + 1]);
            acc += output_offset;
            acc = std::clamp(acc, output_activation_min, output_activation_max);
            local_output[oc + 1] = static_cast<int16_t>(acc);
          }
        }
      }
    }
  }
}

// Optimized for no group, no dilation, 1xn filter.
void conv_per_channel_b64_filter1xn_non_group(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data) {
  const auto batches = MatchingDim(input_shape, 0, output_shape, 0);
  const auto stride_width = params.stride_width;
  const auto pad_width = params.padding_values.width;
  const auto input_width = input_shape.Dims(2);
  const auto input_depth = input_shape.Dims(3);
  const auto input_offset = params.input_offset;
  const auto filter_width = filter_shape.Dims(2);
  const auto filter_depth = filter_shape.Dims(3);
  const auto output_width = output_shape.Dims(2);
  const auto output_depth = output_shape.Dims(3);
  const auto output_offset = params.output_offset;
  const auto output_activation_min = params.quantized_activation_min;
  const auto output_activation_max = params.quantized_activation_max;
  int32_t accumulators[8];
  for (int oc = 0; oc + 2 <= output_depth; oc += 2) {
    for (int batch = 0; batch < batches; ++batch) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = out_x * stride_width - pad_width;

        const int8_t* local_filters0 =
            filter_data + (oc * filter_width * filter_depth);
        const int8_t* local_filters1 =
            local_filters0 + (filter_width * filter_depth);
        const int16_t* local_input = input_data +
            (batch * input_width * input_depth) +
            (in_x_origin * input_depth);
        int16_t* local_output = output_data +
            (batch * output_width * output_depth) +
            (out_x * output_depth);

        vdup_w_x_m(v0, 0);
        vdup_w_x_m(v4, 0);
        ukernel_s8_s16(local_input, local_filters0, local_filters1,
                       filter_width * filter_depth);
        // sum accumulators
        vadd_w_vv(v0, v0, v1);
        vadd_w_vv(v2, v2, v3);
        vadd_w_vv(v0, v0, v2);
        vadd_w_vv(v4, v4, v5);
        vadd_w_vv(v6, v6, v7);
        vadd_w_vv(v4, v4, v6);
        {
          vst_w_x(v0, accumulators);
          int64_t acc64 = bias_data[oc];
          for (int i = 0; i < 8; i++) {
            acc64 += accumulators[i];
          }
          int32_t acc = tflite::MultiplyByQuantizedMultiplier(
                acc64, output_multiplier[oc], output_shift[oc]);
          acc += output_offset;
          acc = std::clamp(acc, output_activation_min, output_activation_max);
          local_output[oc] = static_cast<int16_t>(acc);
        }

        {
          vst_w_x(v4, accumulators);
          int64_t acc64 = bias_data[oc + 1];
          for (int i = 0; i < 8; i++) {
            acc64 += accumulators[i];
          }
          int32_t acc = tflite::MultiplyByQuantizedMultiplier(
                acc64, output_multiplier[oc + 1], output_shift[oc + 1]);
          acc += output_offset;
          acc = std::clamp(acc, output_activation_min, output_activation_max);
          local_output[oc + 1] = static_cast<int16_t>(acc);
        }
      }
    }
  }
}

void conv_per_channel_b64_generic(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data) {
  const auto batches = MatchingDim(input_shape, 0, output_shape, 0);
  const auto stride_width = params.stride_width;
  const auto stride_height = params.stride_height;
  const auto dilation_width_factor = params.dilation_width_factor;
  const auto dilation_height_factor = params.dilation_height_factor;
  const auto pad_width = params.padding_values.width;
  const auto pad_height = params.padding_values.height;
  const auto input_height = input_shape.Dims(1);
  const auto input_width = input_shape.Dims(2);
  const auto input_depth = input_shape.Dims(3);
  const auto input_offset = params.input_offset;
  const auto filter_height = filter_shape.Dims(1);
  const auto filter_width = filter_shape.Dims(2);
  const auto filter_depth = filter_shape.Dims(3);
  const auto output_height = output_shape.Dims(1);
  const auto output_width = output_shape.Dims(2);
  const auto output_depth = output_shape.Dims(3);
  const auto output_offset = params.output_offset;
  const auto output_activation_min = params.quantized_activation_min;
  const auto output_activation_max = params.quantized_activation_max;
  const auto groups = input_depth / filter_depth;
  const auto filters_per_group = output_depth / groups;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = out_y * stride_height - pad_height;
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin = out_x * stride_width - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;
          int64_t acc64 = 0;
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + dilation_width_factor * filter_x;
              const bool inside = (in_x >= 0) && (in_x < input_width) &&
                                  (in_y >= 0) && (in_y < input_height);
              if (!inside) {
                continue;
              }

              int in_channel = 0;
              do {
                int load_count = std::min(filter_depth - in_channel, 16L);
                int32_t input_swizzled[16];
                const int16_t* p_input = &input_data[tflite::Offset(
                    input_shape, batch, in_y, in_x,
                    in_channel + group * filter_depth)];
                for (int i = 0; i < 16; ++i) {
                  int swizzle_idx = swizzle[i];
                  if (swizzle_idx < load_count)
                    input_swizzled[i] = *(p_input + swizzle_idx) + input_offset;
                  else
                    input_swizzled[i] = 0;
                }
                vld_w_l_xx(v0, input_swizzled, 4);
                vld_w_l_xx(v1, input_swizzled + 4, 4);
                vld_w_l_xx(v2, input_swizzled + 8, 4);
                vld_w_l_xx(v3, input_swizzled + 12, 4);
                vld_b_l_xx(v4,
                           &filter_data[tflite::Offset(filter_shape,
                                                       out_channel, filter_y,
                                                       filter_x, in_channel)],
                           load_count);
                vaddw_h_vx(v4, v4, 0);
                vaddw_w_vx(v6, v5, 0);
                vaddw_w_vx(v4, v4, 0);

                vmul_w_vv_m(vm0, vm0, vm1);
                vadd_w_vv(v0, v0, v1);
                vadd_w_vv(v0, v0, v2);
                vadd_w_vv(v0, v0, v3);
                int32_t acc32[4];
                vst_w_l_xx(v0, acc32, 4);
                for (int i = 0; i < 4; ++i) {
                  acc64 += acc32[i];
                }
                in_channel += 16;
              } while (in_channel + 16 <= filter_depth);
            }
          }
          if (bias_data) {
            acc64 = acc64 + bias_data[out_channel];
          }
          int32_t acc = tflite::MultiplyByQuantizedMultiplier(
              acc64, output_multiplier[out_channel], output_shift[out_channel]);
          acc += output_offset;
          acc = std::clamp(acc, output_activation_min, output_activation_max);
          output_data[tflite::Offset(output_shape, batch, out_y, out_x,
                                     out_channel)] = static_cast<int16_t>(acc);
        }
      }
    }
  }
}

void conv_per_channel_b64(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data) {
  if (filter_shape.Dims(kFilterHeightIndex) == 1 &&
      output_shape.Dims(kOutputChannelIndex) % 2 == 0) {
    if (filter_shape.Dims(kFilterWidthIndex) == 1 &&
        filter_shape.Dims(kFilterInputChannelIndex) % 32 == 0) {
      kelvin::opt::conv_per_channel_b64_1x1(
          params, output_multiplier, output_shift, input_shape, input_data,
          filter_shape, filter_data, bias_shape, bias_data, output_shape,
          output_data);
      return;
    }

    // TODO(derekjchow): Check for valid padding
    bool group_conv = !(input_shape.Dims(kInputChannelIndex) ==
        filter_shape.Dims(kFilterInputChannelIndex));
    int32_t fan_in = filter_shape.Dims(kFilterWidthIndex) *
        filter_shape.Dims(kFilterInputChannelIndex);
    if (!group_conv && fan_in % 32 == 0) {
      kelvin::opt::conv_per_channel_b64_filter1xn_non_group(
          params, output_multiplier, output_shift, input_shape, input_data,
          filter_shape, filter_data, bias_shape, bias_data, output_shape,
          output_data);
      return;
    }

    if (fan_in % 32 == 0) {
      kelvin::opt::conv_per_channel_b64_filter1xn_group(
          params, output_multiplier, output_shift, input_shape, input_data,
          filter_shape, filter_data, bias_shape, bias_data, output_shape,
          output_data);
      return;
    }
  }

  kelvin::opt::conv_per_channel_b64_generic(
      params, output_multiplier, output_shift, input_shape, input_data,
      filter_shape, filter_data, bias_shape, bias_data, output_shape,
      output_data);
}

#define INA0 v0
#define FLTA0 v8
#define FLTA1 v9
#define FLTA2 v10
#define FLTA3 v11
#define FLTA4 v12
#define FLTA5 v13
#define FLTA6 v14
#define FLTA7 v15
#define ACC v48
#define ACC0 v48
#define OUT0 v56
void conv_per_channel_b8(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  const auto batches = MatchingDim(input_shape, 0, output_shape, 0);
  const auto stride_width = params.stride_width;
  const auto stride_height = params.stride_height;
  const auto dilation_width_factor = params.dilation_width_factor;
  const auto dilation_height_factor = params.dilation_height_factor;
  const auto pad_width = params.padding_values.width;
  const auto pad_height = params.padding_values.height;
  const auto input_height = input_shape.Dims(1);
  const auto input_width = input_shape.Dims(2);
  const auto input_depth = input_shape.Dims(3);
  const auto input_offset = params.input_offset;
  const auto filter_height = filter_shape.Dims(1);
  const auto filter_width = filter_shape.Dims(2);
  const auto filter_depth = filter_shape.Dims(3);
  const auto output_height = output_shape.Dims(1);
  const auto output_width = output_shape.Dims(2);
  const auto output_depth = output_shape.Dims(3);
  const auto output_offset = params.output_offset;
  const auto output_activation_min = params.quantized_activation_min;
  const auto output_activation_max = params.quantized_activation_max;
  const auto groups = input_depth / filter_depth;
  const auto filters_per_group = output_depth / groups;
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

  // Zero out accumulators.
  vdup_b_x(v0, 0);
  acset_v(ACC, v0);
  vdup_b_x_m(ACC0, 0);
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      const int in_y_origin = (out_y * stride_height) - pad_height;
      for (int out_x = 0; out_x < output_width; /*out_x += 32*/ ++out_x) {
        const int in_x_origin = (out_x * stride_width) - pad_width;
        for (int out_channel = 0; out_channel < output_depth; ++out_channel) {
          auto group = out_channel / filters_per_group;

          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            const int in_x = in_x_origin + dilation_width_factor * 0;

            // Zero padding by omitting the areas outside the image.
            const bool is_point_inside_image =
                (in_x >= 0) && (in_x < input_width) && (in_y >= 0) &&
                (in_y < input_height);
            if (!is_point_inside_image) {
              continue;
            }

            int q = filter_width * filter_depth;
            for (int i = 0; i < q; i += 32) {
              int count = std::min(q - i, 32);
              count = std::min(
                  count, static_cast<int>((input_width - in_x) * filter_depth));
              int input_offset = tflite::Offset(input_shape, batch, in_y, in_x,
                                                group * filter_depth) +
                                 i;
              vdup_w_x_m(vm0, 0);
              vdup_w_x_m(vm1, 0);
              vld_b_l_xx(INA0, &input_data[input_offset], count);
              int filter_offset =
                  tflite::Offset(filter_shape, out_channel, filter_y, 0, 0) + i;
              vdup_w_x_m(FLTA0, 0);
              vdup_w_x_m(FLTA4, 0);
              if (count > 0) {
                vld_b_l_xx(FLTA0, &filter_data[filter_offset],
                           std::min(count, 4));
              }
              if (count > 4) {
                vld_b_l_xx(FLTA1, &filter_data[filter_offset + 4],
                           std::min(count - 4, 4));
              }
              if (count > 8) {
                vld_b_l_xx(FLTA2, &filter_data[filter_offset + 8],
                           std::min(count - 8, 4));
              }
              if (count > 12) {
                vld_b_l_xx(FLTA3, &filter_data[filter_offset + 12],
                           std::min(count - 12, 4));
              }
              if (count > 16) {
                vld_b_l_xx(FLTA4, &filter_data[filter_offset + 16],
                           std::min(count - 16, 4));
              }
              if (count > 20) {
                vld_b_l_xx(FLTA5, &filter_data[filter_offset + 20],
                           std::min(count - 20, 4));
              }
              if (count > 24) {
                vld_b_l_xx(FLTA6, &filter_data[filter_offset + 24],
                           std::min(count - 24, 4));
              }
              if (count > 28) {
                vld_b_l_xx(FLTA7, &filter_data[filter_offset + 28],
                           std::min(count - 28, 4));
              }
              aconv_vxv(ACC, INA0, cmds, FLTA0);
            }
          }
          vcget(ACC);
          vadd_w_vx_m(ACC0, ACC0, bias_data[out_channel]);
          vsll_w_vx_m(ACC0, ACC0, LEFT_SHIFT(output_shift[out_channel]));
          vdmulh_w_r_vx_m(ACC0, ACC0, output_multiplier[out_channel]);
          vsha_w_r_vx_m(ACC0, ACC0, RIGHT_SHIFT(output_shift[out_channel]));
          vadd_w_vx_m(ACC0, ACC0, output_offset);
          vmin_w_vx_m(ACC0, ACC0, output_activation_max);
          vmax_w_vx_m(ACC0, ACC0, output_activation_min);
          vsraqs_b_vx(OUT0, ACC0, 0);
          size_t output_offset =
              tflite::Offset(output_shape, batch, out_y, out_x, out_channel);
          vst_b_l_xx(OUT0, &output_data[output_offset], 1);
        }
      }
    }
  }
}
}  // namespace kelvin::opt
