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

#include "tests/cv/extrema.h"

#include <cstdint>

#include "crt/kelvin.h"

namespace kelvin::cv {

void extrema(int num_cols, const int16_t* input[4][3], uint8_t* output0,
             uint8_t* output1) {
#define prev00 v0
#define prev01 v1
#define prev02 v2
#define prev10 v3
#define prev11 v4
#define prev12 v5
#define prev20 v6
#define prev21 v7
#define prev22 v8
#define prev30 v9
#define prev31 v10
#define prev32 v11
#define curr00 v12
#define curr01 v13
#define curr02 v14
#define curr10 v15
#define curr11 v16
#define curr12 v17
#define curr20 v18
#define curr21 v19
#define curr22 v20
#define curr30 v21
#define curr31 v22
#define curr32 v23
#define next00 v24
#define next01 v25
#define next02 v26
#define next10 v27
#define next11 v28
#define next12 v29
#define next20 v30
#define next21 v31
#define next22 v32
#define next30 v33
#define next31 v34
#define next32 v35
#define elem v36
#define tmin0 v37
#define tmax0 v38
#define tmin1 v39
#define tmax1 v40
#define rmin v41
#define rmax v42
#define value0 v43
#define value1 v44
#define result0 v45
#define result1 v46

  int16_t* ptr0 = const_cast<int16_t*>(input[0][0]);
  int16_t* ptr1 = const_cast<int16_t*>(input[0][1]);
  int16_t* ptr2 = const_cast<int16_t*>(input[0][2]);
  int16_t* ptr3 = const_cast<int16_t*>(input[1][0]);
  int16_t* ptr4 = const_cast<int16_t*>(input[1][1]);
  int16_t* ptr5 = const_cast<int16_t*>(input[1][2]);
  int16_t* ptr6 = const_cast<int16_t*>(input[2][0]);
  int16_t* ptr7 = const_cast<int16_t*>(input[2][1]);
  int16_t* ptr8 = const_cast<int16_t*>(input[2][2]);
  int16_t* ptr9 = const_cast<int16_t*>(input[3][0]);
  int16_t* ptra = const_cast<int16_t*>(input[3][1]);
  int16_t* ptrb = const_cast<int16_t*>(input[3][2]);

  uint8_t* out0 = const_cast<uint8_t*>(output0);
  uint8_t* out1 = const_cast<uint8_t*>(output1);

  vld_h_p_x(curr00, ptr0);
  vld_h_p_x(curr01, ptr1);
  vld_h_p_x(curr02, ptr2);
  vld_h_p_x(curr10, ptr3);
  vld_h_p_x(curr11, ptr4);
  vld_h_p_x(curr12, ptr5);
  vld_h_p_x(curr20, ptr6);
  vld_h_p_x(curr21, ptr7);
  vld_h_p_x(curr22, ptr8);
  vld_h_p_x(curr30, ptr9);
  vld_h_p_x(curr31, ptra);
  vld_h_p_x(curr32, ptrb);

  int vlenh;
  getmaxvl_h(vlenh);

  for (int i = 0; i < num_cols; i += vlenh) {
    // Extrema compute.
#define minmax_p(param0, param1, param2)                            \
  vslidep_h_1_vv(elem, prev##param1##param2, curr##param1##param2); \
  vmin_h_vv(tmin##param0, tmin##param0, elem);                      \
  vmax_h_vv(tmax##param0, tmax##param0, elem);

#define minmax_n(param0, param1, param2)                            \
  vsliden_h_1_vv(elem, curr##param1##param2, next##param1##param2); \
  vmin_h_vv(tmin##param0, tmin##param0, elem);                      \
  vmax_h_vv(tmax##param0, tmax##param0, elem);

#define minmax_c(param0, param1, param2)                       \
  vmin_h_vv(tmin##param0, tmin##param0, prev##param1##param2); \
  vmax_h_vv(tmax##param0, tmax##param0, prev##param1##param2);

    // Common centers.
    vmin_h_vv(tmin0, curr10, curr12);
    vmax_h_vv(tmax0, curr10, curr12);
    vmin_h_vv(tmin0, tmin0, curr20);
    vmax_h_vv(tmax0, tmax0, curr20);
    vmin_h_vv(tmin0, tmin0, curr22);
    vmax_h_vv(tmax0, tmax0, curr22);

    // Common inner two layers.
    vld_h_p_x(next10, ptr3);
    vld_h_p_x(next11, ptr4);
    vld_h_p_x(next12, ptr5);
    minmax_p(0, 1, 0);
    minmax_n(0, 1, 0);
    minmax_p(0, 1, 1);
    minmax_n(0, 1, 1);
    minmax_p(0, 1, 2);
    minmax_n(0, 1, 2);
    vmv_v(prev10, curr10);
    vmv_v(prev11, curr11);
    vmv_v(prev12, curr12);
    vmv_v(curr10, next10);
    vmv_v(curr11, next11);
    vmv_v(curr12, next12);

    vld_h_p_x(next20, ptr6);
    vld_h_p_x(next21, ptr7);
    vld_h_p_x(next22, ptr8);
    minmax_p(0, 2, 0);
    minmax_n(0, 2, 0);
    minmax_p(0, 2, 1);
    minmax_n(0, 2, 1);
    minmax_p(0, 2, 2);
    minmax_n(0, 2, 2);
    vmv_v(prev20, curr20);
    vmv_v(prev21, curr21);
    vmv_v(prev22, curr22);
    vmv_v(curr20, next20);
    vmv_v(curr21, next21);
    vmv_v(curr22, next22);

    // Shared state end.
    vmv_v(tmax1, tmax0);
    vmv_v(tmin1, tmin0);

    // [0,1,2]
    vld_h_p_x(next00, ptr0);
    vld_h_p_x(next01, ptr1);
    vld_h_p_x(next02, ptr2);
    minmax_p(0, 0, 0);
    minmax_n(0, 0, 0);
    minmax_p(0, 0, 1);
    minmax_n(0, 0, 1);
    minmax_p(0, 0, 2);
    minmax_n(0, 0, 2);
    vmv_v(prev00, curr00);
    vmv_v(prev01, curr01);
    vmv_v(prev02, curr02);
    vmv_v(curr00, next00);
    vmv_v(curr01, next01);
    vmv_v(curr02, next02);

    minmax_c(0, 0, 0);
    minmax_c(0, 0, 1);
    minmax_c(0, 0, 2);
    minmax_c(0, 2, 1);

    // [1,2,3]
    vld_h_p_x(next30, ptr9);
    vld_h_p_x(next31, ptra);
    vld_h_p_x(next32, ptrb);
    minmax_p(1, 3, 0);
    minmax_n(1, 3, 0);
    minmax_p(1, 3, 1);
    minmax_n(1, 3, 1);
    minmax_p(1, 3, 2);
    minmax_n(1, 3, 2);
    vmv_v(prev30, curr30);
    vmv_v(prev31, curr31);
    vmv_v(prev32, curr32);
    vmv_v(curr30, next30);
    vmv_v(curr31, next31);
    vmv_v(curr32, next32);

    minmax_c(1, 1, 1);
    minmax_c(1, 3, 0);
    minmax_c(1, 3, 1);
    minmax_c(1, 3, 2);

    // Compare center with min:max.
    vmv_v(value0, prev11);
    vmv_v(value1, prev21);

    vlt_h_vv(rmin, value0, tmin0);
    vgt_h_vv(rmax, value0, tmax0);
    vsll_h_vx(rmax, rmax, 1);
    vor_vv(result0, rmax, rmin);
    vevn_b_vv(result0, result0, result0);
    vst_b_lp_xx(result0, out0, vlenh);

    vlt_h_vv(rmin, value1, tmin1);
    vgt_h_vv(rmax, value1, tmax1);
    vsll_h_vx(rmax, rmax, 1);
    vor_vv(result1, rmax, rmin);
    vevn_b_vv(result1, result1, result1);
    vst_b_lp_xx(result1, out1, vlenh);
  }
}

};  // namespace kelvin::cv
