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

#ifndef TFLM_OPT_OPT_H_
#define TFLM_OPT_OPT_H_

/* clang-format off */
#include <cstring>
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"
/* clang-format on */

namespace kelvin::opt {
void* memcpy(void* dst, const void* src, size_t n);
void elementwise_add_s8(const int8_t* input1, const int8_t* input2,
                        const int32_t input1_offset, const int32_t input1_mult,
                        const int32_t input1_shift, const int32_t input2_offset,
                        const int32_t input2_mult, const int32_t input2_shift,
                        const int32_t left_shift, int8_t* output,
                        const int32_t output_offset, const int32_t output_mult,
                        const int32_t output_shift,
                        const int32_t output_activation_min,
                        const int32_t output_activation_max,
                        const int32_t block_size);
void elementwise_add_s16(const int16_t* input1, const int16_t* input2,
                         const int32_t input1_offset, const int32_t input1_mult,
                         const int32_t input1_shift,
                         const int32_t input2_offset, const int32_t input2_mult,
                         const int32_t input2_shift, const int32_t left_shift,
                         int16_t* output, const int32_t output_offset,
                         const int32_t output_mult, const int32_t output_shift,
                         const int32_t output_activation_min,
                         const int32_t output_activation_max,
                         const int32_t block_size);
void elementwise_add_s32(const int32_t* input1, const int32_t* input2,
                         int32_t* output, const int32_t output_activation_min,
                         const int32_t output_activation_max,
                         const int32_t block_size);
void leaky_relu_s8(const int8_t* input, int8_t* output,
                   const int32_t block_size, const int32_t input_zero_point,
                   const int32_t output_zero_point,
                   const int32_t output_multiplier_alpha,
                   const int32_t output_shift_alpha,
                   const int32_t output_multiplier_identity,
                   const int32_t output_shift_identity);
void leaky_relu_s16(const int16_t* input, int16_t* output,
                    const int32_t block_size, const int32_t input_zero_point,
                    const int32_t output_zero_point,
                    const int32_t output_multiplier_alpha,
                    const int32_t output_shift_alpha,
                    const int32_t output_multiplier_identity,
                    const int32_t output_shift_identity);
void conv_per_channel_b32(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data);

// Top level conv function, will invoke correct variant below.
void conv_per_channel_b64(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data);
void conv_per_channel_b64_1x1(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data);
void conv_per_channel_b64_filter1xn_non_group(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data);
void conv_per_channel_b64_filter1xn_group(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data);
void conv_per_channel_b64_generic(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int16_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int64_t* bias_data, const tflite::RuntimeShape& output_shape,
    int16_t* output_data);

void conv_per_channel_b8(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data);
void DepthwiseConv2DKelvin(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data);
void DWConv2DKelvin_d32(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data);
void DepthwiseConv2DKelvinS16K3x1(
    const int16_t* activations, const int8_t* weights, const int64_t* biases,
    int channels, int frames, int dilation, const int32_t* output_mult,
    const int32_t* output_shift, int32_t output_activation_min,
    int32_t output_activation_max, int16_t* output);
void MaxPoolGeneric(const tflite::PoolParams& params,
                    const tflite::RuntimeShape& input_shape,
                    const int8_t* input_data,
                    const tflite::RuntimeShape& output_shape,
                    int8_t* output_data);

}  // namespace kelvin::opt

#endif  // TFLM_OPT_OPT_H_
