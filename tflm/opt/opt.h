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
void ElementwiseAddS8(const int8_t* input1, const int8_t* input2,
                      const int32_t input1_offset, const int32_t input1_mult,
                      const int32_t input1_shift, const int32_t input2_offset,
                      const int32_t input2_mult, const int32_t input2_shift,
                      const int32_t left_shift, int8_t* output,
                      const int32_t output_offset, const int32_t output_mult,
                      const int32_t output_shift,
                      const int32_t output_activation_min,
                      const int32_t output_activation_max,
                      const int32_t block_size);
void ElementwiseAddS16(const int16_t* input1, const int16_t* input2,
                       const int32_t input1_offset, const int32_t input1_mult,
                       const int32_t input1_shift, const int32_t input2_offset,
                       const int32_t input2_mult, const int32_t input2_shift,
                       const int32_t left_shift, int16_t* output,
                       const int32_t output_offset, const int32_t output_mult,
                       const int32_t output_shift,
                       const int32_t output_activation_min,
                       const int32_t output_activation_max,
                       const int32_t block_size);
void ElementwiseAddS32(const int32_t* input1, const int32_t* input2,
                       int32_t* output, const int32_t output_activation_min,
                       const int32_t output_activation_max,
                       const int32_t block_size);
void LeakyReluS8(const int8_t* input, int8_t* output, const int32_t block_size,
                 const int32_t input_zero_point,
                 const int32_t output_zero_point,
                 const int32_t output_multiplier_alpha,
                 const int32_t output_shift_alpha,
                 const int32_t output_multiplier_identity,
                 const int32_t output_shift_identity);
void LeakyReluS16(const int16_t* input, int16_t* output,
                  const int32_t block_size, const int32_t input_zero_point,
                  const int32_t output_zero_point,
                  const int32_t output_multiplier_alpha,
                  const int32_t output_shift_alpha,
                  const int32_t output_multiplier_identity,
                  const int32_t output_shift_identity);
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

}  // namespace kelvin::opt

#endif  // TFLM_OPT_OPT_H_
