/* Copyright 2020 The TensorFlow Authors. All Rights Reserved.

Licensed under the Apache License, Version 2.0 (the "License");
you may not use this file except in compliance with the License.
You may obtain a copy of the License at

    http://www.apache.org/licenses/LICENSE-2.0

Unless required by applicable law or agreed to in writing, software
distributed under the License is distributed on an "AS IS" BASIS,
WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
See the License for the specific language governing permissions and
limitations under the License.
==============================================================================*/

#include <algorithm>
#include <cmath>

#include "crt/kelvin.h"
#include "tensorflow/lite/kernels/internal/cppmath.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tflm/opt/opt.h"

namespace kelvin::opt {
namespace {
int32_t GetNearestNeighbor(const int input_value, const int32_t input_size,
                           const int32_t output_size, const bool align_corners,
                           const bool half_pixel_centers, const float scale,
                           const float offset) {
  int32_t output_value = std::min(
      align_corners
          ? static_cast<int32_t>(
                tflite::TfLiteRound((input_value + offset) * scale))
          : static_cast<int32_t>(std::floor((input_value + offset) * scale)),
      input_size - 1);
  if (half_pixel_centers) {
    output_value = std::max(static_cast<int32_t>(0), output_value);
  }
  return output_value;
}

void ResizeNN2x(const tflite::ResizeNearestNeighborParams& op_params,
                const tflite::RuntimeShape& input_shape,
                const tflite::RuntimeShape& output_shape,
                const int32_t input_height, const int32_t input_width,
                const int32_t output_height, const int32_t output_width,
                const int8_t* input_data, int8_t* output_data) {
  int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int col_offset = input_shape.Dims(3);
  const int row_offset = input_shape.Dims(2) * col_offset;
  const int batch_offset = input_shape.Dims(1) * row_offset;

  const int8_t* input_ptr = input_data;
  int8_t* output_ptr = output_data;

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < input_height; ++y) {
      const int8_t* input_row_ptr = input_ptr + y * input_width * depth;
      int8_t* output_row_ptr0 = output_ptr + 2 * y * output_width * depth;
      int8_t* output_row_ptr1 = output_row_ptr0 + output_width * depth;

      for (int x = 0; x < input_width; ++x) {
        int channel = 0;
        const int8_t* input_col_ptr = input_row_ptr + x * depth;
        int8_t* output_col_ptr0 = output_row_ptr0 + 2 * x * depth;
        int8_t* output_col_ptr1 = output_row_ptr1 + 2 * x * depth;

        while (channel < depth) {
          vld_b_x(v0, input_col_ptr + channel);
          vst_b_x(v0, output_col_ptr0 + channel);
          vst_b_x(v0, output_col_ptr0 + depth + channel);
          vst_b_x(v0, output_col_ptr1 + channel);
          vst_b_x(v0, output_col_ptr1 + depth + channel);
          channel += 32;
        }
      }
    }
    input_ptr += batch_offset;
  }
}

void ResizeNNGeneric(const tflite::ResizeNearestNeighborParams& op_params,
                     const tflite::RuntimeShape& input_shape,
                     const tflite::RuntimeShape& output_shape,
                     const int32_t input_height, const int32_t input_width,
                     const int32_t output_height, const int32_t output_width,
                     const int8_t* input_data, int8_t* output_data) {
  int32_t batches = MatchingDim(input_shape, 0, output_shape, 0);
  int32_t depth = MatchingDim(input_shape, 3, output_shape, 3);
  const int col_offset = input_shape.Dims(3);
  const int row_offset = input_shape.Dims(2) * col_offset;
  const int batch_offset = input_shape.Dims(1) * row_offset;

  const int8_t* input_ptr = input_data;
  int8_t* output_ptr = output_data;

  const float y_scale =
      (op_params.align_corners && output_height > 1)
          ? (input_height - 1) / static_cast<float>(output_height - 1)
          : input_height / static_cast<float>(output_height);
  const float offset = op_params.half_pixel_centers ? 0.5f : 0.0f;

  const float x_scale =
      (op_params.align_corners && output_width > 1)
          ? (input_width - 1) / static_cast<float>(output_width - 1)
          : input_width / static_cast<float>(output_width);

  for (int b = 0; b < batches; ++b) {
    for (int y = 0; y < output_height; ++y) {
      int32_t in_y = GetNearestNeighbor(
          y, input_height, output_height, op_params.align_corners,
          op_params.half_pixel_centers, y_scale, offset);
      const int8_t* y_input_ptr = input_ptr + in_y * row_offset;
      for (int x = 0; x < output_width; ++x) {
        int32_t in_x = GetNearestNeighbor(
            x, input_width, output_width, op_params.align_corners,
            op_params.half_pixel_centers, x_scale, offset);
        const int8_t* x_input_ptr = y_input_ptr + in_x * col_offset;
        kelvin::opt::Memcpy(output_ptr, x_input_ptr, depth * sizeof(int8_t));

        output_ptr += depth;
      }
    }
    input_ptr += batch_offset;
  }
}
}  // namespace

void ResizeNearestNeighborS8(
    const tflite::ResizeNearestNeighborParams& op_params,
    const tflite::RuntimeShape& unextended_input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& output_size_shape,
    const int32_t* output_size_data,
    const tflite::RuntimeShape& unextended_output_shape, int8_t* output_data) {
  TFLITE_DCHECK_LE(unextended_input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_LE(unextended_output_shape.DimensionsCount(), 4);

  const tflite::RuntimeShape input_shape =
      tflite::RuntimeShape::ExtendedShape(4, unextended_input_shape);
  const tflite::RuntimeShape output_shape =
      tflite::RuntimeShape::ExtendedShape(4, unextended_output_shape);

  int32_t input_height = input_shape.Dims(1);
  int32_t input_width = input_shape.Dims(2);

  TFLITE_DCHECK_EQ(output_size_shape.FlatSize(), 2);
  int32_t output_height = output_size_data[0];
  int32_t output_width = output_size_data[1];

  if (output_height == 2 * input_height && output_width == 2 * input_width) {
    ResizeNN2x(op_params, input_shape, output_shape, input_height, input_width,
               output_height, output_width, input_data, output_data);

  } else {
    ResizeNNGeneric(op_params, input_shape, output_shape, input_height,
                    input_width, output_height, output_width, input_data,
                    output_data);
  }
}

}  // namespace kelvin::opt
