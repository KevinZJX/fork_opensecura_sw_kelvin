// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "tests/cv/upsample.h"

#include <cstdint>

#include "crt/kelvin.h"

namespace kelvin::cv {

void upsample(int num_output_cols, uint16_t* input0_row, uint16_t* input1_row,
              uint16_t* output0_row, uint16_t* output1_row) {
  // Mirror the edges using padded input buffers.
  {
    const int r = (num_output_cols / 2) - 1;
    input0_row[-1] = input0_row[0];
    input1_row[-1] = input1_row[0];
    input0_row[r + 1] = input0_row[r];
    input1_row[r + 1] = input1_row[r];
  }

  int vlenh;
  getmaxvl_h(vlenh);

  uint16_t* __restrict input0 =
      reinterpret_cast<uint16_t*>((input0_row - vlenh));
  uint16_t* __restrict input1 =
      reinterpret_cast<uint16_t*>((input1_row - vlenh));
  uint16_t* __restrict output0 = reinterpret_cast<uint16_t*>(output0_row);
  uint16_t* __restrict output1 = reinterpret_cast<uint16_t*>(output1_row);

#define prev0 v0
#define prev1 v1
#define curr0 v2
#define curr1 v3
#define next0 v4
#define next1 v5
#define c0 curr0
#define c1 curr1
#define p0 v6
#define p0_0 v6
#define p0_1 v7
#define p1 v8
#define p1_0 v8
#define p1_1 v9
#define n0 v10
#define n0_0 v10
#define n0_1 v11
#define n1 v12
#define n1_0 v12
#define n1_1 v13
#define a v14
#define a_0 v14
#define a_1 v15
#define b v16
#define b_0 v16
#define b_1 v17
#define ae v18
#define ae_0 v18
#define ae_1 v19
#define be v20
#define be_0 v20
#define be_1 v21
#define ao v22
#define ao_0 v22
#define ao_1 v23
#define bo v24
#define bo_0 v24
#define bo_1 v25
#define re0 v26
#define re0_0 v26
#define re0_1 v27
#define re1 v28
#define re1_0 v28
#define re1_1 v29
#define ro0 v30
#define ro0_0 v30
#define ro0_1 v31
#define ro1 v32
#define ro1_0 v32
#define ro1_1 v33
#define out0 v34
#define out0_0 v34
#define out0_1 v35
#define out1 v36
#define out1_0 v36
#define out1_1 v37

  vld_h_p_x(prev0, input0);
  vld_h_p_x(prev1, input1);

  vld_h_p_x(curr0, input0);
  vld_h_p_x(curr1, input1);

  for (int i = 0; i < num_output_cols; i += 2 * vlenh) {
    vld_h_p_x(next0, input0);
    vld_h_p_x(next1, input1);

    vslidep_h_1_vv(p0, prev0, curr0);
    vslidep_h_1_vv(p1, prev1, curr1);
    vsliden_h_1_vv(n0, curr0, next0);
    vsliden_h_1_vv(n1, curr1, next1);

    vmulw_w_u_vx(ae, c0, 3);
    vmulw_w_u_vx(be, c1, 3);
    vacc_w_u_vv(ao, ae, n0);
    vacc_w_u_vv(bo, be, n1);
    vacc_w_u_vv(ae, ae, p0);
    vacc_w_u_vv(be, be, p1);

    vmvp_vv(re0, be_0, be_1);
    vmvp_vv(re1, ae_0, ae_1);
    vmvp_vv(ro0, bo_0, bo_1);
    vmvp_vv(ro1, ao_0, ao_1);

    vmacc_w_vx(re0_0, ae_0, 3);
    vmacc_w_vx(re0_1, ae_1, 3);
    vmacc_w_vx(re1_0, be_0, 3);
    vmacc_w_vx(re1_1, be_1, 3);
    vmacc_w_vx(ro0_0, ao_0, 3);
    vmacc_w_vx(ro0_1, ao_1, 3);
    vmacc_w_vx(ro1_0, bo_0, 3);
    vmacc_w_vx(ro1_1, bo_1, 3);

    vsransu_h_r_vx(re0, re0, 4);
    vsransu_h_r_vx(re1, re1, 4);
    vsransu_h_r_vx(ro0, ro0, 4);
    vsransu_h_r_vx(ro1, ro1, 4);

    vzip_h_vv(out0, re0, ro0);
    vzip_h_vv(out1, re1, ro1);
    vst_h_p_x(out0_0, output0);
    vst_h_p_x(out0_1, output0);
    vst_h_p_x(out1_0, output1);
    vst_h_p_x(out1_1, output1);

    vmvp_vv(prev0, curr0, curr1);
    vmvp_vv(curr0, next0, next1);
  }
}

};  // namespace kelvin::cv
