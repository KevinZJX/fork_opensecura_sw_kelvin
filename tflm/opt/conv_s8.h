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

#ifndef TFLM_OPT_CONV_S8_H_
#define TFLM_OPT_CONV_S8_H_

#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"

namespace kelvin::opt {

// filter 1x1 d%32==0
void ConvS8K1x1D32(const tflite::ConvParams& params,
                   const int32_t* output_multiplier, const int32_t* output_shift,
                   const tflite::RuntimeShape& input_shape,
                   const int8_t* input_data,
                   const tflite::RuntimeShape& filter_shape,
                   const int8_t* filter_data,
                   const tflite::RuntimeShape& bias_shape,
                   const int32_t* bias_data,
                   const tflite::RuntimeShape& output_shape, int8_t* output_data);

// filter 1x1 d==16
void ConvS8K1x1D16(const tflite::ConvParams& params,
                   const int32_t* output_multiplier, const int32_t* output_shift,
                   const tflite::RuntimeShape& input_shape,
                   const int8_t* input_data,
                   const tflite::RuntimeShape& filter_shape,
                   const int8_t* filter_data,
                   const tflite::RuntimeShape& bias_shape,
                   const int32_t* bias_data,
                   const tflite::RuntimeShape& output_shape, int8_t* output_data);

// filter depth 4n
void ConvS8D4(const tflite::ConvParams& params,
              const int32_t* output_multiplier, const int32_t* output_shift,
              const tflite::RuntimeShape& input_shape,
              const int8_t* input_data,
              const tflite::RuntimeShape& filter_shape,
              const int8_t* filter_data,
              const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
              const tflite::RuntimeShape& output_shape, int8_t* output_data);

// filter depth 4n, W >= 8
void ConvS8W8D4(const tflite::ConvParams& params,
                const int32_t* output_multiplier, const int32_t* output_shift,
                const tflite::RuntimeShape& input_shape,
                const int8_t* input_data,
                const tflite::RuntimeShape& filter_shape,
                const int8_t* filter_data,
                const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
                const tflite::RuntimeShape& output_shape, int8_t* output_data);

// filter depth 32n
void ConvS8D32(const tflite::ConvParams& params,
               const int32_t* output_multiplier, const int32_t* output_shift,
               const tflite::RuntimeShape& input_shape,
               const int8_t* input_data,
               const tflite::RuntimeShape& filter_shape,
               const int8_t* filter_data,
               const tflite::RuntimeShape& bias_shape, const int32_t* bias_data,
               const tflite::RuntimeShape& output_shape, int8_t* output_data);

// filter size 48x3x1x48
void ConvS8K3x1D48(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data);

// Input depth = 1
void ConvPerChannelD1(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data);

// Input depth = 1, filter_width = 5, filter_height = 5, output_depth = 24
void ConvPerChannelD1OD24_5x5(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data);

}  // namespace kelvin::opt

#endif  // TFLM_OPT_CONV_S8_H_
