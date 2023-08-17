// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef TFLM_OPT_OPT_H_
#define TFLM_OPT_OPT_H_

namespace kelvin::opt {
void *memcpy(void *dst, const void *src, size_t n);
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
}  // namespace kelvin::opt

#endif  // TFLM_OPT_OPT_H_
