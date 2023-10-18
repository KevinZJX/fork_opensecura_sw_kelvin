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

#include "tests/cv/gaussian.h"

#include <cstdio>

#include "crt/kelvin.h"

// Note: separable kernel is vertical then horizontal. H then V with the
// intermediate horizontal is retained may reduce compute further.

namespace kelvin::cv {

static void GaussianVerticalKernel(int num_cols, const uint16_t* input0,
                                   const uint16_t* input1,
                                   const uint16_t* input2,
                                   const uint16_t* input3,
                                   const uint16_t* input4, bool is_stripmine,
                                   uint32_t* output0, uint32_t* output1) {
  uint32_t vl_input, vl_output;
  while (num_cols > 0) {
    if (is_stripmine) {
      getvl_h_x_m(vl_input, num_cols);
      num_cols -= vl_input;
      vl_output = vl_input / 2;
      vld_h_lp_xx_m(vm8, input0, vl_input);
      vld_h_lp_xx_m(vm12, input4, vl_input);
      vld_h_lp_xx_m(vm9, input1, vl_input);
      vld_h_lp_xx_m(vm10, input2, vl_input);
      vld_h_lp_xx_m(vm11, input3, vl_input);

      vaddw_w_u_vv_m(vm0, vm8, vm12);
      vmulw_w_u_vx_m(vm2, vm9, 4);
      vmulw_w_u_vx_m(vm4, vm10, 6);
      vmulw_w_u_vx_m(vm6, vm11, 4);

      vadd3_w_vv_m(vm0, vm2, vm4);
      vadd3_w_vv_m(vm1, vm3, vm5);
      vadd_w_vv_m(vm0, vm0, vm6);
      vadd_w_vv_m(vm1, vm1, vm7);

      vst_w_lp_xx_m(vm0, output0, vl_output);
      vst_w_lp_xx_m(vm1, output1, vl_output);
    } else {
      getvl_h_x(vl_input, num_cols);
      num_cols -= vl_input;
      vl_output = vl_input / 2;
      vld_h_lp_xx(v10, input0, vl_input);
      vld_h_lp_xx(v11, input1, vl_input);
      vld_h_lp_xx(v12, input2, vl_input);
      vld_h_lp_xx(v13, input3, vl_input);
      vld_h_lp_xx(v14, input4, vl_input);

      vaddw_w_u_vv(v16, v10, v14);
      vmulw_w_u_vx(v18, v11, 4);
      vmulw_w_u_vx(v20, v12, 6);
      vmulw_w_u_vx(v22, v13, 4);

      vadd3_w_vv(v16, v18, v20);
      vadd3_w_vv(v17, v19, v21);
      vadd_w_vv(v16, v16, v22);
      vadd_w_vv(v17, v17, v23);

      vst_w_lp_xx(v16, output0, vl_output);
      vst_w_lp_xx(v17, output1, vl_output);
    }
  }
}

static void GaussianHorizontalKernel(int num_cols, const uint32_t* input0,
                                     const uint32_t* input1, bool is_stripmine,
                                     uint16_t* output) {
#define PREV0 vm8
#define PREV1 vm9
#define CURR0 vm10
#define CURR1 vm11
#define NEXT0 vm12
#define NEXT1 vm13
#define P0 vm4
#define P1 vm5
#define N0 vm6
#define N1 vm7
#define SN vm14

#define Rm0 vm0
#define Rm1 vm1
#define R0 v4
#define R1 v5
#define T0 vm2
#define T1 vm3

  uint32_t vl_input, vl_output;

  if (is_stripmine) {
    getmaxvl_w_m(vl_input);

    vld_w_x_m(PREV0, input0 - vl_input);
    vld_w_x_m(PREV1, input1 - vl_input);
    vld_w_p_x_m(CURR0, input0);
    vld_w_p_x_m(CURR1, input1);
  } else {
    getmaxvl_w(vl_input);

    vld_w_x(PREV0, input0 - vl_input);
    vld_w_x(PREV1, input1 - vl_input);
    vld_w_p_x(CURR0, input0);
    vld_w_p_x(CURR1, input1);
  }

  while (num_cols > 0) {
    if (is_stripmine) {
      getvl_h_x_m(vl_output, num_cols);
      num_cols -= vl_output;

      vld_w_p_x_m(NEXT0, input0);
      vld_w_p_x_m(NEXT1, input1);

      vslidehp_w_1_vv_m(P0, PREV0, CURR0);
      vslidehp_w_1_vv_m(P1, PREV1, CURR1);
      vslidehn_w_1_vv_m(N0, CURR0, NEXT0);
      vslidehn_w_1_vv_m(N1, CURR1, NEXT1);

      // even / odd, with additional accumulator
      vmul_w_vx_m(Rm0, P1, 4);
      vmul_w_vx_m(Rm1, CURR0, 4);
      vadd_w_vv_m(T0, P0, N0);
      vadd_w_vv_m(T1, P1, N1);
      vmacc_w_vx_m(Rm0, CURR0, 6);
      vmacc_w_vx_m(Rm1, CURR1, 6);
      vmacc_w_vx_m(T0, CURR1, 4);
      vmacc_w_vx_m(T1, N0, 4);
      vadd_w_vv_m(Rm0, Rm0, T0);
      vadd_w_vv_m(Rm1, Rm1, T1);

      vsransu_h_r_vx_m(SN, Rm0, 8);

      vst_h_lp_xx_m(SN, output, vl_output);

      vmv_v_m(PREV0, CURR0);
      vmv_v_m(PREV1, CURR1);
      vmv_v_m(CURR0, NEXT0);
      vmv_v_m(CURR1, NEXT1);
    } else {
      getvl_h_x(vl_output, num_cols);
      num_cols -= vl_output;

      vld_w_p_x(NEXT0, input0);
      vld_w_p_x(NEXT1, input1);

      vslidep_w_1_vv(P0, PREV0, CURR0);
      vslidep_w_1_vv(P1, PREV1, CURR1);
      vsliden_w_1_vv(N0, CURR0, NEXT0);
      vsliden_w_1_vv(N1, CURR1, NEXT1);

      // even
      vadd_w_vv(R0, P0, N0);
      vmacc_w_vx(R0, P1, 4);
      vmacc_w_vx(R0, CURR0, 6);
      vmacc_w_vx(R0, CURR1, 4);

      // odd
      vadd_w_vv(R1, P1, N1);
      vmacc_w_vx(R1, CURR0, 4);
      vmacc_w_vx(R1, CURR1, 6);
      vmacc_w_vx(R1, N0, 4);

      vsransu_h_r_vx(SN, R0, 8);

      vst_h_lp_xx(SN, output, vl_output);

      vmv_v(PREV0, CURR0);
      vmv_v(PREV1, CURR1);
      vmv_v(CURR0, NEXT0);
      vmv_v(CURR1, NEXT1);
    }
  }
}

#define ARRAYSIZE(x) sizeof(x) / sizeof(x[0])

void gaussian(int num_cols, const uint16_t* input0_row,
              const uint16_t* input1_row, const uint16_t* input2_row,
              const uint16_t* input3_row, const uint16_t* input4_row,
              bool is_stripmine, uint16_t* output_row) {
  int vlenw;
  getmaxvl_w(vlenw);
  const int interleave_num = num_cols / 2 - 1;  // even/odd interleaved
  uint32_t temp0_data_unpadded[1024 + 2 * vlenw] __attribute__((aligned(64)));
  uint32_t temp1_data_unpadded[1024 + 2 * vlenw] __attribute__((aligned(64)));
  uint32_t* temp0_data = temp0_data_unpadded + vlenw;
  uint32_t* temp1_data = temp1_data_unpadded + vlenw;

  GaussianVerticalKernel(num_cols, input0_row, input1_row, input2_row,
                         input3_row, input4_row, is_stripmine, temp0_data,
                         temp1_data);
  if (temp0_data <= temp0_data_unpadded ||
      ((temp0_data - temp0_data_unpadded) / sizeof(uint32_t) + interleave_num +
           1 >=
       ARRAYSIZE(temp0_data_unpadded))) {
    printf("**error**: temp0_data out of bound\n");
    exit(1);
  }
  if (temp1_data <= temp1_data_unpadded ||
      ((temp1_data - temp1_data_unpadded) / sizeof(uint32_t) + interleave_num +
           1 >=
       ARRAYSIZE(temp1_data_unpadded))) {
    printf("**error**: temp1_data out of bound\n");
    exit(1);
  }
  temp0_data[-1] = temp0_data[0];
  temp1_data[-1] = temp0_data[0];
  temp0_data[interleave_num + 1] = temp1_data[interleave_num];
  temp1_data[interleave_num + 1] = temp1_data[interleave_num];
  GaussianHorizontalKernel(num_cols, temp0_data, temp1_data, is_stripmine,
                           output_row);
}

};  // namespace kelvin::cv
