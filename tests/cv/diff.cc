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

#include "tests/cv/diff.h"

#include <cstdint>

#include "crt/kelvin.h"

#define likely(x) __builtin_expect(!!(x), 1)
#define unlikely(x) __builtin_expect(!!(x), 0)

namespace kelvin::cv {

void diff(int num_cols, const uint16_t* input0_row, const uint16_t* input1_row,
          uint16_t* output_row) {
  int vl;
  int n = num_cols;
  do {
    getvl_h_x_m(vl, n);
    n -= vl;
    vld_h_lp_xx_m(vm1, input0_row, vl);
    vld_h_lp_xx_m(vm2, input1_row, vl);
    vsub_h_vv_m(vm0, vm1, vm2);
    vst_h_lp_xx_m(vm0, output_row, vl);
  } while (n > 0);
}

void diffp(int num_cols, const uint16_t* input0_row, const uint16_t* input1_row,
           uint16_t* output_row) {
  int vl_0, vl_1;
  int n = num_cols;

  // [0] load
  getvl_h_x_m(vl_0, n);
  n -= vl_0;
  vld_h_lp_xx_m(v4, input0_row, vl_0);
  vld_h_lp_xx_m(v8, input1_row, vl_0);

  while (true) {
    // [1] load
    getvl_h_x_m(vl_1, n);
    n -= vl_1;
    vld_h_lp_xx_m(v20, input0_row, vl_1);
    vld_h_lp_xx_m(v24, input1_row, vl_1);

    // [0] store
    vsub_h_vv_m(v0, v4, v8);
    vst_h_lp_xx_m(v0, output_row, vl_0);
    if (unlikely(!vl_1)) break;

    // [0] load
    getvl_h_x_m(vl_0, n);
    n -= vl_0;
    vld_h_lp_xx_m(v4, input0_row, vl_0);
    vld_h_lp_xx_m(v8, input1_row, vl_0);

    // [1] store
    vsub_h_vv_m(v16, v20, v24);
    vst_h_lp_xx_m(v16, output_row, vl_1);
    if (unlikely(!vl_0)) break;
  }
}

void diff4(int num_cols, int stride, const uint16_t* input0_row,
           const uint16_t* input1_row, uint16_t* output_row) {
  int vl;
  int n = num_cols;
  do {
    getvl_h_x(vl, n);
    n -= vl;
    vld_h_tp_xx_m(v4, input0_row, stride);
    vld_h_tp_xx_m(v8, input1_row, stride);
    vsub_h_vv_m(v0, v4, v8);
    vst_h_tp_xx_m(v0, output_row, stride);
  } while (n > 0);
}

void diff4p(int num_cols, int stride, const uint16_t* input0_row,
            const uint16_t* input1_row, uint16_t* output_row) {
  int vl_0, vl_1;
  int n = num_cols;

  // [0] load
  getvl_h_x(vl_0, n);
  n -= vl_0;
  vld_h_tp_xx_m(v4, input0_row, stride);
  vld_h_tp_xx_m(v8, input1_row, stride);

  while (true) {
    // [1] load
    getvl_h_x(vl_1, n);
    n -= vl_1;
    if (likely(vl_1)) {
      vld_h_tp_xx_m(v20, input0_row, stride);
      vld_h_tp_xx_m(v24, input1_row, stride);
    }

    // [0] store
    vsub_h_vv_m(v0, v4, v8);
    vst_h_tp_xx_m(v0, output_row, stride);
    if (unlikely(!vl_1)) break;

    // [0] load
    getvl_h_x(vl_0, n);
    n -= vl_0;
    if (likely(vl_0)) {
      vld_h_tp_xx_m(v4, input0_row, stride);
      vld_h_tp_xx_m(v8, input1_row, stride);
    }

    // [1] store
    vsub_h_vv_m(v16, v20, v24);
    vst_h_tp_xx_m(v16, output_row, stride);
    if (unlikely(!vl_0)) break;
  }
}

};  // namespace kelvin::cv
