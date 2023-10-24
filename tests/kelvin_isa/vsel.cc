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

#include <cstdio>

#include "tests/kelvin_isa/kelvin_test.h"

#define vsel_vv(T, Vd, Vs, Vt) \
  {                            \
    if (sizeof(T) == 1) {      \
      vsel_b_vv(Vd, Vs, Vt);   \
    }                          \
    if (sizeof(T) == 2) {      \
      vsel_h_vv(Vd, Vs, Vt);   \
    }                          \
    if (sizeof(T) == 4) {      \
      vsel_w_vv(Vd, Vs, Vt);   \
    }                          \
  }

#define vsel_vv_m(T, Vd, Vs, Vt) \
  {                              \
    if (sizeof(T) == 1) {        \
      vsel_b_vv_m(Vd, Vs, Vt);   \
    }                            \
    if (sizeof(T) == 2) {        \
      vsel_h_vv_m(Vd, Vs, Vt);   \
    }                            \
    if (sizeof(T) == 4) {        \
      vsel_w_vv_m(Vd, Vs, Vt);   \
    }                            \
  }

template <typename T>
static void test_vsel() {
  constexpr int n = sizeof(T);
  int lanes;
  getmaxvl(T, lanes);

  T inp[3][lanes] __attribute__((aligned(64)));
  T ref[lanes] __attribute__((aligned(64)));
  T dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    inp[0][i] = (0x40 << (8 * (n - 1))) + i;
    inp[1][i] = (0x80 << (8 * (n - 1))) + i;
    inp[2][i] = krand();
  }

  for (int i = 0; i < lanes; ++i) {
    ref[i] = inp[2][i] & 1 ? inp[0][i] : inp[1][i];
  }

  vld_x(T, v0, inp[0]);
  vld_x(T, v1, inp[1]);
  vld_x(T, v2, inp[2]);
  vsel_vv(T, v0, v2, v1);
  vst_x(T, v0, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vsel[%d]\n", i);
      exit(-1);
    }
  }
}

template <typename T>
static void test_vsel_m() {
  constexpr int n = sizeof(T);
  int lanes;
  getmaxvl_m(T, lanes);

  T inp[3][lanes * 2] __attribute__((aligned(64)));
  T ref[lanes] __attribute__((aligned(64)));
  T dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    inp[0][i] = (0x40 << (8 * (n - 1))) + i;
    inp[1][i] = (0x80 << (8 * (n - 1))) + i;
    inp[2][i] = krand();
  }

  for (int i = 0; i < lanes; ++i) {
    ref[i] = inp[2][i] & 1 ? inp[0][i] : inp[1][i];
  }

  vld_x_m(T, v0, inp[0]);
  vld_x_m(T, v4, inp[1]);
  vld_x_m(T, v8, inp[2]);
  vsel_vv_m(T, v0, v8, v4);
  vst_x_m(T, v0, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vsel_m[%d]\n", i);
      exit(-1);
    }
  }
}

int main() {
  test_vsel<int8_t>();
  test_vsel<int16_t>();
  test_vsel<int32_t>();

  test_vsel_m<int8_t>();
  test_vsel_m<int16_t>();
  test_vsel_m<int32_t>();

  return 0;
}
