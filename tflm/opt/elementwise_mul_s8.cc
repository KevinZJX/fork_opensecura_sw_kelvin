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
void MulS8(const tflite::ArithmeticParams& params,
           const tflite::RuntimeShape& input1_shape, const int8_t* input1_data,
           const tflite::RuntimeShape& input2_shape, const int8_t* input2_data,
           const tflite::RuntimeShape& output_shape, int8_t* output_data) {
  const int32_t input1_offset = params.input1_offset;
  const int32_t input2_offset = params.input2_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_mult = params.output_multiplier;
  const int32_t output_shift = params.output_shift;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  const int block_size =
      MatchingElementsSize(input1_shape, input2_shape, output_shape);

  int blocks = block_size;

  while (blocks >= 96) {
    vld_b_lp_xx(v0, input1_data, 32);
    vld_b_lp_xx(v8, input2_data, 32);

    vaddw_h_vx(v2, v0, 0);
    vaddw_h_vx(v10, v8, 0);
    vaddw_w_vx(v4, v2, input1_offset);
    vaddw_w_vx(v6, v3, input1_offset);
    vaddw_w_vx(v12, v10, input2_offset);
    vaddw_w_vx(v14, v11, input2_offset);

    vld_b_lp_xx(v16, input1_data, 32);
    vld_b_lp_xx(v24, input2_data, 32);

    vaddw_h_vx(v18, v16, 0);
    vaddw_h_vx(v26, v24, 0);
    vaddw_w_vx(v20, v18, input1_offset);
    vaddw_w_vx(v22, v19, input1_offset);
    vaddw_w_vx(v28, v26, input2_offset);
    vaddw_w_vx(v30, v27, input2_offset);

    vld_b_lp_xx(v32, input1_data, 32);
    vld_b_lp_xx(v40, input2_data, 32);

    vaddw_h_vx(v34, v32, 0);
    vaddw_h_vx(v42, v40, 0);
    vaddw_w_vx(v36, v34, input1_offset);
    vaddw_w_vx(v38, v35, input1_offset);
    vaddw_w_vx(v44, v42, input2_offset);
    vaddw_w_vx(v46, v43, input2_offset);

    vmul_w_vv_m(v12, v4, v12);
    vmul_w_vv_m(v28, v20, v28);
    vmul_w_vv_m(v44, v36, v44);

    vdmulh_w_r_vx_m(v12, v12, output_mult);
    vdmulh_w_r_vx_m(v28, v28, output_mult);
    vdmulh_w_r_vx_m(v44, v44, output_mult);
    vsha_w_r_vx_m(v12, v12, -output_shift);
    vsha_w_r_vx_m(v28, v28, -output_shift);
    vsha_w_r_vx_m(v44, v44, -output_shift);
    vadd_w_vx_m(v12, v12, output_offset);
    vadd_w_vx_m(v28, v28, output_offset);
    vadd_w_vx_m(v44, v44, output_offset);

    vmin_w_vx_m(v12, v12, output_activation_max);
    vmin_w_vx_m(v28, v28, output_activation_max);
    vmin_w_vx_m(v44, v44, output_activation_max);
    vmax_w_vx_m(v12, v12, output_activation_min);
    vmax_w_vx_m(v28, v28, output_activation_min);
    vmax_w_vx_m(v44, v44, output_activation_min);

    vsraqs_b_vx(v12, v12, 0);
    vst_b_lp_xx(v12, output_data, 32);
    vsraqs_b_vx(v28, v28, 0);
    vst_b_lp_xx(v28, output_data, 32);
    vsraqs_b_vx(v44, v44, 0);
    vst_b_lp_xx(v44, output_data, 32);

    blocks -= 96;
  }

  while (blocks) {
    int count = std::min(blocks, 32);
    vld_b_lp_xx(v0, input1_data, count);
    vld_b_lp_xx(v8, input2_data, count);

    vaddw_h_vx(v2, v0, 0);
    vaddw_h_vx(v10, v8, 0);
    vaddw_w_vx(v4, v2, input1_offset);
    vaddw_w_vx(v6, v3, input1_offset);
    vaddw_w_vx(v12, v10, input2_offset);
    vaddw_w_vx(v14, v11, input2_offset);

    vmul_w_vv_m(v16, v4, v12);

    rescale_m(v16, v16, output_mult, output_shift, output_offset);

    vmin_w_vx_m(v16, v16, output_activation_max);
    vmax_w_vx_m(v16, v16, output_activation_min);

    vsraqs_b_vx(v16, v16, 0);
    vst_b_lp_xx(v16, output_data, count);

    blocks -= count;
  }
}

}  // namespace kelvin::opt
