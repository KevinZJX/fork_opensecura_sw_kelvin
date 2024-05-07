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
// Data types: input: s16, filter: s8, bias s64

#include "tflm/opt/conv_util.h"

namespace kelvin::opt {
namespace {
// Accumulates in v0-v7. [v0-v3], [v4-v7] are sub accumulators for two outputs.
// Load/swizzle filters use [v52-v63].
// Input activations use [v32-v33].
// No clobbers.
void ConvUkernelS8S16(const int16_t* input_data0, const int8_t* filter_data0,
                      const int8_t* filter_data1, size_t n) {
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

void ConvS16B64K1x1(
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
        ConvUkernelS8S16(group_input, local_filters0, local_filters1,
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
void ConvS16B64K1xnGroup(
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
          const int16_t* local_input =
              input_data + (b * input_width * input_depth) +
              (in_x_origin * input_depth) + (g * filter_depth);
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

            ConvUkernelS8S16(local_inputx, local_filters0x, local_filters1x,
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
void ConvS16B64K1xnNonGroup(
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
        ConvUkernelS8S16(local_input, local_filters0, local_filters1,
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

void ConvS16B64Generic(
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
}  // namespace

void ConvS16B64(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data) {
  const auto input_depth = input_shape.Dims(3);
  const auto filter_height = filter_shape.Dims(1);
  const auto filter_width = filter_shape.Dims(2);
  const auto filter_depth = filter_shape.Dims(3);
  const auto output_depth = output_shape.Dims(3);

  // generic implementation by default
  auto fn = ConvS16B64Generic;

  // special cases
  if (filter_height == 1 && output_depth % 2 == 0) {
    // 1x1 filter, filter depth = 32n
    if (filter_width == 1 && filter_depth % 32 == 0) {
      fn = ConvS16B64K1x1;
    }

    // 1xn non group filter
    bool group_conv = !(input_depth == filter_depth);
    int32_t fan_in = filter_width * filter_depth;
    if (!group_conv && fan_in % 32 == 0) {
      fn = ConvS16B64K1xnNonGroup;
    }

    // 1xn group filter
    if (fan_in % 32 == 0) {
      fn = ConvS16B64K1xnGroup;
    }
  }

  fn(params, output_multiplier, output_shift, input_shape, input_data,
     filter_shape, filter_data, bias_shape, bias_data, output_shape,
     output_data);
}

}  // namespace kelvin::opt
