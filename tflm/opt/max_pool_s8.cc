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

#include "crt/kelvin.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"

namespace kelvin::opt {
void MaxPoolGeneric(const tflite::PoolParams &params,
                    const tflite::RuntimeShape &input_shape,
                    const int8_t *input_data,
                    const tflite::RuntimeShape &output_shape,
                    int8_t *output_data) {
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int stride_height = params.stride_height;
  const int stride_width = params.stride_width;
  for (int batch = 0; batch < batches; ++batch) {
    for (int out_y = 0; out_y < output_height; ++out_y) {
      for (int out_x = 0; out_x < output_width; ++out_x) {
        const int in_x_origin =
            (out_x * stride_width) - params.padding_values.width;
        const int in_y_origin =
            (out_y * stride_height) - params.padding_values.height;

        // Compute the boundaries of the filter region clamped so as to
        // ensure that the filter window fits in the input array.
        const int filter_x_start = std::max(0, -in_x_origin);
        const int filter_x_end =
            std::min(params.filter_width, input_width - in_x_origin);
        const int filter_y_start = std::max(0, -in_y_origin);
        const int filter_y_end =
            std::min(params.filter_height, input_height - in_y_origin);

        int channel = 0;
        for (; channel + 32 <= depth; channel += 32) {
          vdup_b_x(v0, params.quantized_activation_min);
          for (int filter_y = filter_y_start; filter_y < filter_y_end;
               ++filter_y) {
            for (int filter_x = filter_x_start; filter_x < filter_x_end;
                 ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              const int in_y = in_y_origin + filter_y;
              const int8_t *local_input =
                  input_data + Offset(input_shape, batch, in_y, in_x, channel);
              vld_b_x(v1, local_input);
              vmax_b_vv(v0, v0, v1);
            }
          }
          vmin_b_vx(v0, v0, params.quantized_activation_max);
          int8_t *local_output =
              output_data + Offset(output_shape, batch, out_y, out_x, channel);
          vst_b_x(v0, local_output);
        }

        if (channel == depth) {
          continue;
        }
        int remaining_channels = depth - channel;
        vdup_b_x(v0, params.quantized_activation_min);
        for (int filter_y = filter_y_start; filter_y < filter_y_end;
             ++filter_y) {
          for (int filter_x = filter_x_start; filter_x < filter_x_end;
               ++filter_x) {
            const int in_x = in_x_origin + filter_x;
            const int in_y = in_y_origin + filter_y;
            const int8_t *local_input =
                input_data + Offset(input_shape, batch, in_y, in_x, depth - 1);
            vld_b_l_xx(v1, local_input, remaining_channels);
            vmax_b_vv(v0, v0, v1);
          }
        }
        vmin_b_vx(v0, v0, params.quantized_activation_max);
        int8_t *local_output =
            output_data + Offset(output_shape, batch, out_y, out_x, depth - 1);
        vst_b_l_xx(v0, local_output, remaining_channels);
      }
    }
  }
}

} // namespace kelvin::opt
