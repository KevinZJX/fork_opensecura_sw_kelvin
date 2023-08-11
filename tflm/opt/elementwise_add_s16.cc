// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "crt/kelvin.h"
#include "tflm/opt/opt.h"
#include "tflm/opt/util.h"

namespace kelvin::opt {

void elementwise_add_s16(const int16_t* input1, const int16_t* input2,
                         const int32_t input1_offset, const int32_t input1_mult,
                         const int32_t input1_shift,
                         const int32_t input2_offset, const int32_t input2_mult,
                         const int32_t input2_shift, const int32_t left_shift,
                         int16_t* output, const int32_t output_offset,
                         const int32_t output_mult, const int32_t output_shift,
                         const int32_t output_activation_min,
                         const int32_t output_activation_max,
                         const int32_t block_size) {
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

    rescale_m(vm0, vm0, input1_mult, input1_shift, input1_offset);
    rescale_m(vm1, vm1, input1_mult, input1_shift, input1_offset);
    rescale_m(vm2, vm2, input2_mult, input2_shift, input2_offset);
    rescale_m(vm3, vm3, input2_mult, input2_shift, input2_offset);

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
