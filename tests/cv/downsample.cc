// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "tests/cv/downsample.h"

#include <cstdint>

#include "crt/kelvin.h"

namespace kelvin::cv {

void downsample(int num_output_cols, const uint16_t* input0_row,
                const uint16_t* input1_row, uint16_t* output_row) {
  int vl_input_0, vl_input_1, vl_output;
  int m = 2 * num_output_cols;
  int n = num_output_cols;
  while (n > 0) {
    getvl_h_x_m(vl_input_0, m);
    m -= vl_input_0;
    getvl_h_x_m(vl_input_1, m);
    m -= vl_input_1;
    getvl_h_x_m(vl_output, n);
    n -= vl_output;

    vld_h_lp_xx_m(vm12, input0_row, vl_input_0);
    vld_h_lp_xx_m(vm13, input0_row, vl_input_1);
    vld_h_lp_xx_m(vm14, input1_row, vl_input_0);
    vld_h_lp_xx_m(vm15, input1_row, vl_input_1);

    vpadd_w_u_v_m(vm8, vm12);
    vpadd_w_u_v_m(vm9, vm13);
    vpadd_w_u_v_m(vm10, vm14);
    vpadd_w_u_v_m(vm11, vm15);
    vadd_w_vv_m(vm6, vm8, vm10);
    vadd_w_vv_m(vm7, vm9, vm11);

    vevnodd_w_vv_m(vm4, vm6, vm7);

    vsransu_h_r_vx_m(vm0, vm4, 2);

    vst_h_lp_xx_m(vm0, output_row, vl_output);
  }
}

};  // namespace kelvin::cv
