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

#ifndef TFLM_OPT_OPT_H_
#define TFLM_OPT_OPT_H_

/* clang-format off */
#include <cstring>
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"
/* clang-format on */

namespace kelvin::opt {
void* Memcpy(void* dst, const void* src, size_t n);
void ElementwiseAddS8(const tflite::ArithmeticParams& params,
                      const tflite::RuntimeShape& input1_shape,
                      const int8_t* input1_data,
                      const tflite::RuntimeShape& input2_shape,
                      const int8_t* input2_data,
                      const tflite::RuntimeShape& output_shape,
                      int8_t* output_data);
void ElementwiseAddS16(const tflite::ArithmeticParams& params,
                       const tflite::RuntimeShape& input1_shape,
                       const int16_t* input1_data,
                       const tflite::RuntimeShape& input2_shape,
                       const int16_t* input2_data,
                       const tflite::RuntimeShape& output_shape,
                       int16_t* output_data);
void ElementwiseAddS32(const tflite::ArithmeticParams& params,
                       const tflite::RuntimeShape& input1_shape,
                       const int32_t* input1_data,
                       const tflite::RuntimeShape& input2_shape,
                       const int32_t* input2_data,
                       const tflite::RuntimeShape& output_shape,
                       int32_t* output_data);
void LeakyReluS8(const tflite::LeakyReluParams& params,
                 const tflite::RuntimeShape& input_shape,
                 const int8_t* input_data,
                 const tflite::RuntimeShape& output_shape, int8_t* output_data);
void LeakyReluS16(const tflite::LeakyReluParams& params,
                  const tflite::RuntimeShape& input_shape,
                  const int16_t* input_data,
                  const tflite::RuntimeShape& output_shape,
                  int16_t* output_data);
void ConvS16B32(const tflite::ConvParams& params,
                const int32_t* output_multiplier, const int32_t* output_shift,
                const tflite::RuntimeShape& input_shape,
                const int16_t* input_data,
                const tflite::RuntimeShape& filter_shape,
                const int8_t* filter_data,
                const tflite::RuntimeShape& bias_shape,
                const int32_t* bias_data,
                const tflite::RuntimeShape& output_shape, int16_t* output_data);
void ConvS16B64(const tflite::ConvParams& params,
                const int32_t* output_multiplier, const int32_t* output_shift,
                const tflite::RuntimeShape& input_shape,
                const int16_t* input_data,
                const tflite::RuntimeShape& filter_shape,
                const int8_t* filter_data,
                const tflite::RuntimeShape& bias_shape,
                const int64_t* bias_data,
                const tflite::RuntimeShape& output_shape, int16_t* output_data);
void ConvS8(const tflite::ConvParams& params, const int32_t* output_multiplier,
            const int32_t* output_shift,
            const tflite::RuntimeShape& input_shape, const int8_t* input_data,
            const tflite::RuntimeShape& filter_shape, const int8_t* filter_data,
            const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
            const tflite::RuntimeShape& output_shape, int8_t* output_data);
void DepthwiseConvS8(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data);
void DepthwiseConvS16(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data);
void MaxPoolS8(const tflite::PoolParams& params,
               const tflite::RuntimeShape& input_shape,
               const int8_t* input_data,
               const tflite::RuntimeShape& output_shape, int8_t* output_data);
void LogisticS8(int32_t input_zero_point, int32_t input_range_radius,
                int32_t input_multiplier, int32_t input_left_shift,
                int32_t input_size, const int8_t* input_data,
                int8_t* output_data);
void ResizeNearestNeighborS8(
    const tflite::ResizeNearestNeighborParams& op_params,
    const tflite::RuntimeShape& unextended_input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& output_size_shape,
    const int32_t* output_size_data,
    const tflite::RuntimeShape& unextended_output_shape, int8_t* output_data);

}  // namespace kelvin::opt

#endif  // TFLM_OPT_OPT_H_
