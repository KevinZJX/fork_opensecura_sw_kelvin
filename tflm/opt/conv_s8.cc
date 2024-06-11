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

#include "tflm/opt/conv_s8.h"

#include <algorithm>

#include "tensorflow/lite/kernels/internal/reference/integer_ops/conv.h"
#include "tflm/opt/conv_util.h"

namespace kelvin::opt {
namespace {
void ConvS8Generic(
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

  if (pad_width > 0 || pad_height > 0 || dilation_width_factor > 1 ||
      dilation_height_factor > 1) {
    // use reference implementation
    tflite::reference_integer_ops::ConvPerChannel(
        params, output_multiplier, output_shift, input_shape, input_data,
        filter_shape, filter_data, bias_shape, bias_data, output_shape,
        output_data);
    return;
  }

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
}  // namespace

void ConvS8(const tflite::ConvParams& params, const int32_t* output_multiplier,
            const int32_t* output_shift,
            const tflite::RuntimeShape& input_shape, const int8_t* input_data,
            const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
            const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
            const tflite::RuntimeShape& output_shape, int8_t* output_data) {
  const auto batches = MatchingDim(input_shape, 0, output_shape, 0);
  const auto stride_width = params.stride_width;
  const auto stride_height = params.stride_height;
  const auto dilation_width_factor = params.dilation_width_factor;
  const auto dilation_height_factor = params.dilation_height_factor;
  const auto pad_width = params.padding_values.width;
  const auto pad_height = params.padding_values.height;
  const auto input_width = input_shape.Dims(2);
  const auto input_depth = input_shape.Dims(3);
  const auto filter_height = filter_shape.Dims(1);
  const auto filter_width = filter_shape.Dims(2);
  const auto filter_depth = filter_shape.Dims(3);
  const auto output_width = output_shape.Dims(2);
  const auto output_depth = output_shape.Dims(3);

#define RUN_KERNEL(kernel) {\
  kernel(\
      params, output_multiplier, output_shift, input_shape, input_data,\
      filter_shape, filter_data, bias_shape, bias_data, output_shape,\
      output_data);\
  return; \
}

  // special case of filter size 1x1
  if (filter_height == 1 && filter_width == 1 && stride_height == 1 &&
      stride_width == 1 && dilation_height_factor == 1 &&
      dilation_width_factor == 1 && pad_height == 0 && pad_width == 0 &&
      (input_depth == filter_depth)) {
    if ((output_depth % 8) == 0 && (input_depth % 32) == 0) {
      RUN_KERNEL(kelvin::opt::ConvS8K1x1D32);
    }

    // TODO: Relax this kernel for all output_depths
    if ((output_depth < 8) && (input_depth % 32) == 0) {
      RUN_KERNEL(kelvin::opt::ConvS8K1x1D32);
    }

    if ((output_depth % 16) == 0 && (input_depth == 16)) {
      RUN_KERNEL(kelvin::opt::ConvS8K1x1D16);
    }
  }

  if (input_depth == 1 && filter_width == 5 && filter_height == 5 &&
      output_depth == 24) {
    RUN_KERNEL(kelvin::opt::ConvPerChannelD1OD24_5x5);
  }

  // special case of filter_depth = 4n, stride 2 and min width
  if (dilation_width_factor == 1 && dilation_height_factor == 1 &&
      stride_width == 2 && stride_height == 2 && filter_depth % 4 == 0 &&
      output_depth >= 8 && output_width >= 8 && pad_width <= 1) {
    RUN_KERNEL(kelvin::opt::ConvS8W8D4);
  }

  // special case of filter_depth = 4n
  if (dilation_width_factor == 1 && dilation_height_factor == 1 &&
      stride_width <= 2 && stride_height <= 2 && filter_depth % 4 == 0 &&
      output_depth >= 8 && output_width >= 8 && pad_width <= 1) {
    RUN_KERNEL(kelvin::opt::ConvS8D4);
  }

  // special case of filter depth = 32n
  if (dilation_width_factor == 1 && dilation_height_factor == 1 &&
      stride_width <= 2 && stride_height <= 2 && filter_depth % 32 == 0) {
    RUN_KERNEL(kelvin::opt::ConvS8D32);
  }

  // special case of filter size 48x3x1x48
  if (batches == 1 && filter_height == 3 && filter_width == 1 &&
      input_width == 1 && input_depth == 48 && output_depth == 48 &&
      stride_height == 1 && stride_width == 1 && dilation_height_factor == 1 &&
      dilation_width_factor == 1 && pad_height == 0 && pad_width == 0) {
    RUN_KERNEL(kelvin::opt::ConvS8K3x1D48);
  }

  if (input_depth == 1 && ((output_depth % 4) == 0)) {
    RUN_KERNEL(kelvin::opt::ConvPerChannelD1);
  }

  RUN_KERNEL(ConvS8Generic);
}

}  // namespace kelvin::opt
