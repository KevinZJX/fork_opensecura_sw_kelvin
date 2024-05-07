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

#include <limits>

#include "crt/kelvin.h"
#include "tflm/opt/opt.h"
#include "tflm/opt/util.h"

namespace kelvin::opt {
void LeakyReluS16(const int16_t* input, int16_t* output,
                  const int32_t block_size, const int32_t input_zero_point,
                  const int32_t output_zero_point,
                  const int32_t output_multiplier_alpha,
                  const int32_t output_shift_alpha,
                  const int32_t output_multiplier_identity,
                  const int32_t output_shift_identity) {
  constexpr int32_t quantized_output_min = std::numeric_limits<int16_t>::min();
  constexpr int32_t quantized_output_max = std::numeric_limits<int16_t>::max();
  int32_t right_shift_identity = std::min(output_shift_identity, 0L);
  int32_t left_shift_identity = std::max(output_shift_identity, 0L);
  int32_t right_shift_alpha = std::min(output_shift_alpha, 0L);
  int32_t left_shift_alpha = std::max(output_shift_alpha, 0L);
  int blocks = block_size;
  int vl;
  getmaxvl_h(vl);
  while (blocks) {
    int count = std::min(blocks, vl);

    // Load data from the input, and widen.
    vld_h_lp_xx(v0, input, count);
    vaddw_w_vx(v0, v0, 0);

    // Subtract out the provided offset from the inputs.
    vsub_w_vx_m(vm0, vm0, input_zero_point);

    // Compute the Relu on all inputs, as if they were >=0.
    vsll_w_vx_m(vm2, vm0, left_shift_identity);
    vdmulh_w_r_vx_m(vm2, vm2, output_multiplier_identity);
    vsha_w_vx_m(vm2, vm2, RIGHT_SHIFT(right_shift_identity));
    vadd_w_vx_m(vm2, vm2, output_zero_point);
    vmax_w_vx_m(vm2, vm2, quantized_output_min);
    vmin_w_vx_m(vm2, vm2, quantized_output_max);

    // Compute the Relu on all inputs, as if they were <0.
    vsll_w_vx_m(vm1, vm0, left_shift_alpha);
    vdmulh_w_r_vx_m(vm1, vm1, output_multiplier_alpha);
    vsha_w_vx_m(vm1, vm1, RIGHT_SHIFT(right_shift_alpha));
    vadd_w_vx_m(vm1, vm1, output_zero_point);
    vmax_w_vx_m(vm1, vm1, quantized_output_min);
    vmin_w_vx_m(vm1, vm1, quantized_output_max);

    // Compute a boolean vector for inputs >=0.
    vge_w_vx_m(vm3, vm0, 0);
    // Compute a boolean vector for inputs <0.
    vlt_w_vx_m(vm0, vm0, 0);
    // Multiply the `identity` results by the >=0 vector.
    vmul_w_vv_m(vm2, vm2, vm3);
    // Multiply the `alpha` results by the <0 vector.
    vmul_w_vv_m(vm0, vm1, vm0);
    // Sum the two resulting vectors.
    vadd_w_vv_m(vm0, vm0, vm2);

    // Narrow/swizzle, and store to output.
    vsrans_h_vx(v0, v0, 0);
    vst_h_lp_xx(v0, output, count);

    blocks -= count;
  }
}

}  // namespace kelvin::opt
