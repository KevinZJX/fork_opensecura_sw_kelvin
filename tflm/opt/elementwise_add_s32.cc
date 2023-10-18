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

namespace kelvin::opt {
void elementwise_add_s32(const int32_t* input1, const int32_t* input2,
                         int32_t* output, const int32_t output_activation_min,
                         const int32_t output_activation_max,
                         const int32_t block_size) {
  int blocks = block_size;
  int vl;
  getmaxvl_w_m(vl);
  while (blocks) {
    int count = std::min(blocks, vl);

    vld_w_p_xx_m(vm0, input1, count);
    vld_w_p_xx_m(vm1, input2, count);

    vadd_w_vv_m(vm0, vm0, vm1);
    vmin_w_vx_m(vm0, vm0, output_activation_max);
    vmax_w_vx_m(vm0, vm0, output_activation_min);

    vst_w_p_xx_m(vm0, output, count);

    blocks -= count;
  }
}
}  // namespace kelvin::opt
