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

#include "crt/printf_traits.h"
#include "tests/kelvin_isa/kelvin_test.h"

#define vzip_vv(T, Vd, Vs, Vt) \
  {                            \
    if (sizeof(T) == 1) {      \
      vzip_b_vv(Vd, Vs, Vt);   \
    }                          \
    if (sizeof(T) == 2) {      \
      vzip_h_vv(Vd, Vs, Vt);   \
    }                          \
    if (sizeof(T) == 4) {      \
      vzip_w_vv(Vd, Vs, Vt);   \
    }                          \
  }

#define vzip_vv_m(T, Vd, Vs, Vt) \
  {                              \
    if (sizeof(T) == 1) {        \
      vzip_b_vv_m(Vd, Vs, Vt);   \
    }                            \
    if (sizeof(T) == 2) {        \
      vzip_h_vv_m(Vd, Vs, Vt);   \
    }                            \
    if (sizeof(T) == 4) {        \
      vzip_w_vv_m(Vd, Vs, Vt);   \
    }                            \
  }

template <typename T>
static void test_vzip() {
  constexpr int n = sizeof(T);
  int lanes;
  getmaxvl(T, lanes);

  T inp[2][lanes] __attribute__((aligned(64)));
  T ref[2][lanes] __attribute__((aligned(64)));
  T dut[2][lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    inp[0][i] = (0x40 << (8 * (n - 1))) + i;
    inp[1][i] = (0x80 << (8 * (n - 1))) + i;
  }

  for (int i = 0; i < lanes; ++i) {
    int m = i / 2;
    int n = i / 2 + lanes / 2;
    ref[0][i] = i & 1 ? inp[1][m] : inp[0][m];
    ref[1][i] = i & 1 ? inp[1][n] : inp[0][n];
  }

  vld_x(T, v2, inp[0]);
  vld_x(T, v3, inp[1]);
  vzip_vv(T, v0, v2, v3);
  vst_x(T, v0, dut[0]);
  vst_x(T, v1, dut[1]);

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < lanes; ++i) {
      if (ref[j][i] != dut[j][i]) {
        printf("**error vzip_vv[%d,%d] ", j, i);
        printf(PrintfTraits<T>::kFmtHex, ref[j][i]);
        printf(" ");
        printf(PrintfTraits<T>::kFmtHex, dut[j][i]);
        printf("\n");
        exit(-1);
      }
    }
  }
}

template <typename T>
static void test_vzip_m() {
  constexpr int n = sizeof(T);
  int lanes;
  getmaxvl_m(T, lanes);

  T inp[2][1024] __attribute__((aligned(64)));
  T ref[2][1024] __attribute__((aligned(64)));
  T dut[2][1024] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    inp[0][i] = (0x40 << (8 * (n - 1))) + i;
    inp[1][i] = (0x80 << (8 * (n - 1))) + i;
  }

  for (int i = 0; i < lanes; ++i) {
    int m = i / 2;
    int n = i / 2 + lanes / 2;
    ref[0][i] = i & 1 ? inp[1][m] : inp[0][m];
    ref[1][i] = i & 1 ? inp[1][n] : inp[0][n];
  }

  vld_x_m(T, v8, inp[0]);
  vld_x_m(T, v12, inp[1]);
  vzip_vv_m(T, v0, v8, v12);
  vst_x_m(T, v0, dut[0]);
  vst_x_m(T, v4, dut[1]);

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < lanes; ++i) {
      if (ref[j][i] != dut[j][i]) {
        printf("**error vzip_vv_m[%d,%d] ", j, i);
        printf(PrintfTraits<T>::kFmtHex, ref[j][i]);
        printf(" ");
        printf(PrintfTraits<T>::kFmtHex, dut[j][i]);
        printf("\n");
        exit(-1);
      }
    }
  }
}

int main() {
  test_vzip<int8_t>();
  test_vzip<int16_t>();
  test_vzip<int32_t>();

  test_vzip_m<int8_t>();
  test_vzip_m<int16_t>();
  test_vzip_m<int32_t>();

  return 0;
}
