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

#include "crt/kelvin.h"
#include "tflm/opt/opt.h"
#include "tflm/opt/util.h"

namespace kelvin::opt {

void ElementwiseAddS16(const tflite::ArithmeticParams& params,
                       const tflite::RuntimeShape& input1_shape,
                       const int16_t* input1,
                       const tflite::RuntimeShape& input2_shape,
                       const int16_t* input2,
                       const tflite::RuntimeShape& output_shape,
                       int16_t* output) {
  const int32_t input1_offset = params.input1_offset;
  const int32_t input1_mult = params.input1_multiplier;
  const int32_t input1_shift = params.input1_shift;
  const int32_t input2_offset = params.input2_offset;
  const int32_t input2_mult = params.input2_multiplier;
  const int32_t input2_shift = params.input2_shift;
  const int32_t left_shift = params.left_shift;
  const int32_t output_offset = params.output_offset;
  const int32_t output_mult = params.output_multiplier;
  const int32_t output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  const int block_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  int blocks = block_size;
  int vl;
  getmaxvl_h(vl);
  while (blocks) {
    int count = std::min(blocks, vl);

    // Widen input1 to 32-bit wide values (in vm0, vm1).
    vld_h_lp_xx_m(vm0, input1, count);
    vaddw_w_vx_m(vm0, vm0, input1_offset);

    // Widen input2 to 32-bit wide values (in vm2, vm3).
    vld_h_lp_xx_m(vm2, input2, count);
    vaddw_w_vx_m(vm2, vm2, input2_offset);

    // Apply left_shift to all inputs.
    vsll_w_vx_m(vm0, vm0, left_shift);
    vsll_w_vx_m(vm1, vm1, left_shift);
    vsll_w_vx_m(vm2, vm2, left_shift);
    vsll_w_vx_m(vm3, vm3, left_shift);

    int32_t input1_shift_mul = 1 << LEFT_SHIFT(input1_shift);
    int32_t input2_shift_mul = 1 << LEFT_SHIFT(input2_shift);
    vmul_w_vx_m(vm0, vm0, input1_shift_mul);
    vmul_w_vx_m(vm1, vm1, input1_shift_mul);
    vmul_w_vx_m(vm2, vm2, input2_shift_mul);
    vmul_w_vx_m(vm3, vm3, input2_shift_mul);

    rescale_m(vm0, vm0, input1_mult, input1_shift, 0);
    rescale_m(vm1, vm1, input1_mult, input1_shift, 0);
    rescale_m(vm2, vm2, input2_mult, input2_shift, 0);
    rescale_m(vm3, vm3, input2_mult, input2_shift, 0);

    // Sum the rescaled inputs.
    vadd_w_vv_m(vm0, vm0, vm2);
    vadd_w_vv_m(vm1, vm1, vm3);

    // Rescale the summed output.
    rescale_m(vm0, vm0, output_mult, output_shift, output_offset);
    rescale_m(vm1, vm1, output_mult, output_shift, output_offset);

    // Clamp to the provided range.
    vmin_w_vx_m(vm0, vm0, output_activation_max);
    vmin_w_vx_m(vm1, vm1, output_activation_max);
    vmax_w_vx_m(vm0, vm0, output_activation_min);
    vmax_w_vx_m(vm1, vm1, output_activation_min);

    // Swizzle and narrow back to bytes.
    vand_w_vx_m(vm0, vm0, 0xFFFF);
    vand_w_vx_m(vm1, vm1, 0xFFFF);
    vsll_w_vx_m(vm1, vm1, 16);
    vor_vv_m(vm0, vm0, vm1);

    // Store to memory.
    vst_h_lp_xx_m(vm0, output, count);

    blocks -= count;
  }
}

}  // namespace kelvin::opt
