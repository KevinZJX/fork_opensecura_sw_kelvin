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
#include "tflm/opt/opt.h"
#include "tflm/opt/util.h"

namespace kelvin::opt {

void elementwise_add_s8(const int8_t* input1, const int8_t* input2,
                        const int32_t input1_offset, const int32_t input1_mult,
                        const int32_t input1_shift, const int32_t input2_offset,
                        const int32_t input2_mult, const int32_t input2_shift,
                        const int32_t left_shift, int8_t* output,
                        const int32_t output_offset, const int32_t output_mult,
                        const int32_t output_shift,
                        const int32_t output_activation_min,
                        const int32_t output_activation_max,
                        const int32_t block_size) {
  int blocks = block_size;
  int vl;
  getmaxvl_b(vl);

  const int32_t input1_shift_mul = 1 << LEFT_SHIFT(input1_shift);
  const int32_t input2_shift_mul = 1 << LEFT_SHIFT(input2_shift);

  while (blocks) {
    int count = std::min(blocks, vl);

    // Widen input1 to 32-bit wide values (in vm0, vm1, vm2, vm3).
    vld_b_lp_xx_m(vm0, input1, count);
    vaddw_h_vx_m(vm0, vm0, 0);
    vaddw_w_vx_m(vm2, vm1, input1_offset);
    vaddw_w_vx_m(vm0, vm0, input1_offset);

    // Widen input2 to 32-bit wide values (in vm4, vm5, vm6, vm7).
    vld_b_lp_xx_m(vm4, input2, count);
    vaddw_h_vx_m(vm4, vm4, 0);
    vaddw_w_vx_m(vm6, vm5, input2_offset);
    vaddw_w_vx_m(vm4, vm4, input2_offset);

    // Apply left_shift to all inputs.
    vsll_w_vx_m(vm0, vm0, left_shift);
    vsll_w_vx_m(vm1, vm1, left_shift);
    vsll_w_vx_m(vm2, vm2, left_shift);
    vsll_w_vx_m(vm3, vm3, left_shift);
    vsll_w_vx_m(vm4, vm4, left_shift);
    vsll_w_vx_m(vm5, vm5, left_shift);
    vsll_w_vx_m(vm6, vm6, left_shift);
    vsll_w_vx_m(vm7, vm7, left_shift);

    vmul_w_vx_m(vm0, vm0, input1_shift_mul);
    vmul_w_vx_m(vm1, vm1, input1_shift_mul);
    vmul_w_vx_m(vm2, vm2, input1_shift_mul);
    vmul_w_vx_m(vm3, vm3, input1_shift_mul);
    vmul_w_vx_m(vm4, vm4, input2_shift_mul);
    vmul_w_vx_m(vm5, vm5, input2_shift_mul);
    vmul_w_vx_m(vm6, vm6, input2_shift_mul);
    vmul_w_vx_m(vm7, vm7, input2_shift_mul);

    rescale_m(vm0, vm0, vm15, input1_mult, input1_shift, input1_offset);
    rescale_m(vm1, vm1, vm15, input1_mult, input1_shift, input1_offset);
    rescale_m(vm2, vm2, vm15, input1_mult, input1_shift, input1_offset);
    rescale_m(vm3, vm3, vm15, input1_mult, input1_shift, input1_offset);
    rescale_m(vm4, vm4, vm15, input2_mult, input2_shift, input2_offset);
    rescale_m(vm5, vm5, vm15, input2_mult, input2_shift, input2_offset);
    rescale_m(vm6, vm6, vm15, input2_mult, input2_shift, input2_offset);
    rescale_m(vm7, vm7, vm15, input2_mult, input2_shift, input2_offset);

    // Sum the rescaled inputs.
    vadd_w_vv_m(vm0, vm0, vm4);
    vadd_w_vv_m(vm1, vm1, vm5);
    vadd_w_vv_m(vm2, vm2, vm6);
    vadd_w_vv_m(vm3, vm3, vm7);

    // Rescale the summed output.
    rescale_m(vm0, vm0, vm15, output_mult, output_shift, output_offset);
    rescale_m(vm1, vm1, vm15, output_mult, output_shift, output_offset);
    rescale_m(vm2, vm2, vm15, output_mult, output_shift, output_offset);
    rescale_m(vm3, vm3, vm15, output_mult, output_shift, output_offset);

    // Clamp to the provided range.
    vmin_w_vx_m(vm0, vm0, output_activation_max);
    vmin_w_vx_m(vm1, vm1, output_activation_max);
    vmin_w_vx_m(vm2, vm2, output_activation_max);
    vmin_w_vx_m(vm3, vm3, output_activation_max);
    vmax_w_vx_m(vm0, vm0, output_activation_min);
    vmax_w_vx_m(vm1, vm1, output_activation_min);
    vmax_w_vx_m(vm2, vm2, output_activation_min);
    vmax_w_vx_m(vm3, vm3, output_activation_min);

    // Swizzle and narrow back to bytes.
    vsraqs_b_vx_m(vm0, vm0, 0);

    // Store to memory.
    vst_b_lp_xx_m(vm0, output, count);

    blocks -= count;
  }
}

}  // namespace kelvin::opt
