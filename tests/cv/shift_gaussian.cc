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

#include "tests/cv/shift_gaussian.h"

#include <cstdint>
#include <cstdio>

#include "crt/kelvin.h"

// Note: separable kernel is vertical then horizontal. H then V with the
// intermediate horizontal is retained may reduce compute further.

namespace kelvin::cv {

static void GaussianVerticalKernel(int num_cols, const uint8_t* input0,
                                   const uint8_t* input1, const uint8_t* input2,
                                   const uint8_t* input3, const uint8_t* input4,
                                   bool is_stripmine, uint16_t* output) {
  uint32_t vl_input, vl_output_0, vl_output_1;
  while (num_cols > 0) {
    if (is_stripmine) {
      getvl_b_x_m(vl_input, num_cols);
      getvl_h_x_m(vl_output_0, num_cols);
      num_cols -= vl_input;
      vl_output_1 = vl_input - vl_output_0;

      vld_b_lp_xx_m(v32, input0, vl_input);
      vld_b_lp_xx_m(v36, input1, vl_input);
      vld_b_lp_xx_m(v40, input2, vl_input);
      vld_b_lp_xx_m(v44, input3, vl_input);
      vld_b_lp_xx_m(v48, input4, vl_input);

      vaddw_h_u_vv_m(v0, v32, v48);
      vmulw_h_u_vx_m(v8, v36, 4);
      vmulw_h_u_vx_m(v16, v40, 6);
      vmulw_h_u_vx_m(v24, v44, 4);

      vadd3_h_vv_m(v0, v8, v16);
      vadd3_h_vv_m(v4, v12, v20);
      vadd_h_vv_m(v0, v0, v24);
      vadd_h_vv_m(v4, v4, v28);

      vzip_h_vv_m(v16, v0, v4);

      vst_h_lp_xx_m(v16, output, vl_output_0);
      vst_h_lp_xx_m(v20, output, vl_output_1);
    } else {
      getvl_b_x(vl_input, num_cols);
      getvl_h_x(vl_output_0, num_cols);
      num_cols -= vl_input;
      vl_output_1 = vl_input - vl_output_0;

      vld_b_lp_xx(v10, input0, vl_input);
      vld_b_lp_xx(v11, input1, vl_input);
      vld_b_lp_xx(v12, input2, vl_input);
      vld_b_lp_xx(v13, input3, vl_input);
      vld_b_lp_xx(v14, input4, vl_input);

      vaddw_h_u_vv(v16, v10, v14);
      vmulw_h_u_vx(v18, v11, 4);
      vmulw_h_u_vx(v20, v12, 6);
      vmulw_h_u_vx(v22, v13, 4);

      vadd3_h_vv(v16, v18, v20);
      vadd3_h_vv(v17, v19, v21);
      vadd_h_vv(v16, v16, v22);
      vadd_h_vv(v17, v17, v23);

      vzip_h_vv(v0, v16, v17);

      vst_h_lp_xx(v0, output, vl_output_0);
      vst_h_lp_xx(v1, output, vl_output_1);
    }
  }
}

static void GaussianHorizontalKernel(int num_cols, const uint16_t* input,
                                     bool is_stripmine, uint16_t* output) {
#define PREV v32
#define CURR v40
#define NEXT v48
#define P2 v16
#define P1 v20
#define N1 v24
#define N2 v28
#define RS v0

  uint32_t vl_input, vl_output;

  if (is_stripmine) {
    getmaxvl_h_m(vl_input);

    vld_h_x_m(PREV, input - vl_input);
    vld_h_p_x_m(CURR, input);
  } else {
    getmaxvl_h(vl_input);

    vld_h_x(PREV, input - vl_input);
    vld_h_p_x(CURR, input);
  }

  while (num_cols > 0) {
    if (is_stripmine) {
      getvl_h_x_m(vl_output, num_cols);
      num_cols -= vl_output;

      vld_h_p_x_m(NEXT, input);

      vslidehp_h_2_vv_m(P2, PREV, CURR);
      vslidehp_h_1_vv_m(P1, PREV, CURR);
      vslidehn_h_1_vv_m(N1, CURR, NEXT);
      vslidehn_h_2_vv_m(N2, CURR, NEXT);

      vadd_h_vv_m(RS, P2, N2);
      vmacc_h_vx_m(RS, P1, 4);
      vmacc_h_vx_m(RS, CURR, 6);
      vmacc_h_vx_m(RS, N1, 4);

      vst_h_lp_xx_m(RS, output, vl_output);

      vmv_v_m(PREV, CURR);
      vmv_v_m(CURR, NEXT);
    } else {
      getvl_h_x(vl_output, num_cols);
      num_cols -= vl_output;

      vld_h_p_x(NEXT, input);

      vslidep_h_2_vv(P2, PREV, CURR);
      vslidep_h_1_vv(P1, PREV, CURR);
      vsliden_h_1_vv(N1, CURR, NEXT);
      vsliden_h_2_vv(N2, CURR, NEXT);

      vadd_h_vv(RS, P2, N2);
      vmacc_h_vx(RS, P1, 4);
      vmacc_h_vx(RS, CURR, 6);
      vmacc_h_vx(RS, N1, 4);

      vst_h_lp_xx(RS, output, vl_output);

      vmv_v(PREV, CURR);
      vmv_v(CURR, NEXT);
    }
  }
}

void shift_gaussian(int num_cols, const uint8_t* input0_row,
                    const uint8_t* input1_row, const uint8_t* input2_row,
                    const uint8_t* input3_row, const uint8_t* input4_row,
                    bool is_stripmine, uint16_t* output_row) {
  int vlenh;
  getmaxvl_h(vlenh);
  const int r = num_cols - 1;
  uint16_t temp_data_unpadded[1024 + 2 * vlenh] __attribute__((aligned(64)));
  uint16_t* temp_data = temp_data_unpadded + vlenh;

  GaussianVerticalKernel(num_cols, input0_row, input1_row, input2_row,
                         input3_row, input4_row, is_stripmine, temp_data);
  if (temp_data <= &temp_data_unpadded[1]) {
    printf("**error**: temp_data out of bound\n");
    exit(1);
  }
  temp_data[-1] = temp_data[0];
  temp_data[-2] = temp_data[0];
  temp_data[r + 1] = temp_data[r];
  temp_data[r + 2] = temp_data[r];
  GaussianHorizontalKernel(num_cols, temp_data, is_stripmine, output_row);
}

};  // namespace kelvin::cv
