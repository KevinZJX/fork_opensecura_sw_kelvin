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

#define vevn_vv(T, Vd, Vs, Vt) \
  {                            \
    if (sizeof(T) == 1) {      \
      vevn_b_vv(Vd, Vs, Vt);   \
    }                          \
    if (sizeof(T) == 2) {      \
      vevn_h_vv(Vd, Vs, Vt);   \
    }                          \
    if (sizeof(T) == 4) {      \
      vevn_w_vv(Vd, Vs, Vt);   \
    }                          \
  }

#define vevn_vv_m(T, Vd, Vs, Vt) \
  {                              \
    if (sizeof(T) == 1) {        \
      vevn_b_vv_m(Vd, Vs, Vt);   \
    }                            \
    if (sizeof(T) == 2) {        \
      vevn_h_vv_m(Vd, Vs, Vt);   \
    }                            \
    if (sizeof(T) == 4) {        \
      vevn_w_vv_m(Vd, Vs, Vt);   \
    }                            \
  }

#define vodd_vv(T, Vd, Vs, Vt) \
  {                            \
    if (sizeof(T) == 1) {      \
      vodd_b_vv(Vd, Vs, Vt);   \
    }                          \
    if (sizeof(T) == 2) {      \
      vodd_h_vv(Vd, Vs, Vt);   \
    }                          \
    if (sizeof(T) == 4) {      \
      vodd_w_vv(Vd, Vs, Vt);   \
    }                          \
  }

#define vodd_vv_m(T, Vd, Vs, Vt) \
  {                              \
    if (sizeof(T) == 1) {        \
      vodd_b_vv_m(Vd, Vs, Vt);   \
    }                            \
    if (sizeof(T) == 2) {        \
      vodd_h_vv_m(Vd, Vs, Vt);   \
    }                            \
    if (sizeof(T) == 4) {        \
      vodd_w_vv_m(Vd, Vs, Vt);   \
    }                            \
  }

#define vevnodd_vv(T, Vd, Vs, Vt) \
  {                               \
    if (sizeof(T) == 1) {         \
      vevnodd_b_vv(Vd, Vs, Vt);   \
    }                             \
    if (sizeof(T) == 2) {         \
      vevnodd_h_vv(Vd, Vs, Vt);   \
    }                             \
    if (sizeof(T) == 4) {         \
      vevnodd_w_vv(Vd, Vs, Vt);   \
    }                             \
  }

#define vevnodd_vv_m(T, Vd, Vs, Vt) \
  {                                 \
    if (sizeof(T) == 1) {           \
      vevnodd_b_vv_m(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 2) {           \
      vevnodd_h_vv_m(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 4) {           \
      vevnodd_w_vv_m(Vd, Vs, Vt);   \
    }                               \
  }

template <typename T>
static void test_vevnodd() {
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
    int m = lanes / 2;
    ref[0][i] = i < m ? inp[0][2 * i + 0] : inp[1][2 * (i - m) + 0];
    ref[1][i] = i < m ? inp[0][2 * i + 1] : inp[1][2 * (i - m) + 1];
  }

  vld_x(T, v2, inp[0]);
  vld_x(T, v7, inp[1]);
  vevnodd_vv(T, v0, v2, v7);
  vst_x(T, v0, dut[0]);
  vst_x(T, v1, dut[1]);

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < lanes; ++i) {
      if (ref[j][i] != dut[j][i]) {
        printf("**error vevnodd_vv[%d,%d] %x %x\n", j, i, ref[j][i], dut[j][i]);
        exit(-1);
      }
    }
  }
}

template <typename T>
static void test_vevnodd_m() {
  constexpr int n = sizeof(T);
  int lanes;
  getmaxvl_m(T, lanes);

  T inp[2][lanes] __attribute__((aligned(64)));
  T ref[2][lanes] __attribute__((aligned(64)));
  T dut[2][lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    inp[0][i] = (0x40 << (8 * (n - 1))) + i;
    inp[1][i] = (0x80 << (8 * (n - 1))) + i;
  }

  for (int i = 0; i < lanes; ++i) {
    int m = lanes / 2;
    ref[0][i] = i < m ? inp[0][2 * i + 0] : inp[1][2 * (i - m) + 0];
    ref[1][i] = i < m ? inp[0][2 * i + 1] : inp[1][2 * (i - m) + 1];
  }

  vld_x_m(T, v8, inp[0]);
  vld_x_m(T, v24, inp[1]);
  vevnodd_vv_m(T, v0, v8, v24);
  vst_x_m(T, v0, dut[0]);
  vst_x_m(T, v4, dut[1]);

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < lanes; ++i) {
      if (ref[j][i] != dut[j][i]) {
        printf("**error vevnodd_vv_m[%d,%d] %x %x\n", j, i, ref[j][i],
               dut[j][i]);
        exit(-1);
      }
    }
  }
}

template <typename T>
static void test_vevn() {
  constexpr int n = sizeof(T);
  int lanes;
  getmaxvl(T, lanes);

  T inp[2][lanes] __attribute__((aligned(64)));
  T ref[lanes] __attribute__((aligned(64)));
  T dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    inp[0][i] = (0x40 << (8 * (n - 1))) + i;
    inp[1][i] = (0x80 << (8 * (n - 1))) + i;
  }

  for (int i = 0; i < lanes; ++i) {
    int m = lanes / 2;
    ref[i] = i < m ? inp[0][2 * i + 0] : inp[1][2 * (i - m) + 0];
  }

  vld_x(T, v2, inp[0]);
  vld_x(T, v7, inp[1]);
  vevn_vv(T, v0, v2, v7);
  vst_x(T, v0, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vevn_vv[%d] %x %x\n", i, ref[i], dut[i]);
      exit(-1);
    }
  }
}

template <typename T>
static void test_vevn_m() {
  constexpr int n = sizeof(T);
  int lanes;
  getmaxvl_m(T, lanes);

  T inp[2][lanes] __attribute__((aligned(64)));
  T ref[lanes] __attribute__((aligned(64)));
  T dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    inp[0][i] = (0x40 << (8 * (n - 1))) + i;
    inp[1][i] = (0x80 << (8 * (n - 1))) + i;
  }

  for (int i = 0; i < lanes; ++i) {
    int m = lanes / 2;
    ref[i] = i < m ? inp[0][2 * i + 0] : inp[1][2 * (i - m) + 0];
  }

  vld_x_m(T, v4, inp[0]);
  vld_x_m(T, v24, inp[1]);
  vevn_vv_m(T, v0, v4, v24);
  vst_x_m(T, v0, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vevn_vv_m[%d] %x %x\n", i, ref[i], dut[i]);
      exit(-1);
    }
  }
}

template <typename T>
static void test_vodd() {
  constexpr int n = sizeof(T);
  int lanes;
  getmaxvl(T, lanes);

  T inp[2][lanes] __attribute__((aligned(64)));
  T ref[lanes] __attribute__((aligned(64)));
  T dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    inp[0][i] = (0x40 << (8 * (n - 1))) + i;
    inp[1][i] = (0x80 << (8 * (n - 1))) + i;
  }

  for (int i = 0; i < lanes; ++i) {
    int m = lanes / 2;
    ref[i] = i < m ? inp[0][2 * i + 1] : inp[1][2 * (i - m) + 1];
  }

  vld_x(T, v2, inp[0]);
  vld_x(T, v7, inp[1]);
  vodd_vv(T, v0, v2, v7);
  vst_x(T, v0, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vodd_vv[%d] %x %x\n", i, ref[i], dut[i]);
      exit(-1);
    }
  }
}

template <typename T>
static void test_vodd_m() {
  constexpr int n = sizeof(T);
  int lanes;
  getmaxvl_m(T, lanes);

  T inp[2][lanes] __attribute__((aligned(64)));
  T ref[lanes] __attribute__((aligned(64)));
  T dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    inp[0][i] = (0x40 << (8 * (n - 1))) + i;
    inp[1][i] = (0x80 << (8 * (n - 1))) + i;
  }

  for (int i = 0; i < lanes; ++i) {
    int m = lanes / 2;
    ref[i] = i < m ? inp[0][2 * i + 1] : inp[1][2 * (i - m) + 1];
  }

  // TODO(b/307456144): Re-enable {v4, v24} as the inputs.
  vld_x_m(T, v12, inp[0]);
  vld_x_m(T, v24, inp[1]);
  vodd_vv_m(T, v0, v12, v24);
  vst_x_m(T, v0, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vodd_vv_m[%d] %x %x\n", i, ref[i], dut[i]);
      exit(-1);
    }
  }
}

int main() {
  test_vevnodd<int8_t>();
  test_vevnodd<int16_t>();
  test_vevnodd<int32_t>();

  test_vevnodd_m<int8_t>();
  test_vevnodd_m<int16_t>();
  test_vevnodd_m<int32_t>();

  test_vevn<int8_t>();
  test_vevn<int16_t>();
  test_vevn<int32_t>();

  test_vevn_m<int8_t>();
  test_vevn_m<int16_t>();
  test_vevn_m<int32_t>();

  test_vodd<int8_t>();
  test_vodd<int16_t>();
  test_vodd<int32_t>();

  test_vodd_m<int8_t>();
  test_vodd_m<int16_t>();
  test_vodd_m<int32_t>();

  return 0;
}
