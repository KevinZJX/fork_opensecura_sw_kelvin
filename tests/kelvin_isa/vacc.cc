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

#define vacc_v(T, Vd, Vs, Vt)       \
  {                                 \
    if (sizeof(T) == 2) {           \
      if (std::is_signed<T>::value) \
        vacc_h_vv(Vd, Vs, Vt);      \
      else                          \
        vacc_h_u_vv(Vd, Vs, Vt);    \
    }                               \
    if (sizeof(T) == 4) {           \
      if (std::is_signed<T>::value) \
        vacc_w_vv(Vd, Vs, Vt);      \
      else                          \
        vacc_w_u_vv(Vd, Vs, Vt);    \
    }                               \
  }

#define vacc_v(T, Vd, Vs, Vt)       \
  {                                 \
    if (sizeof(T) == 2) {           \
      if (std::is_signed<T>::value) \
        vacc_h_vv(Vd, Vs, Vt);      \
      else                          \
        vacc_h_u_vv(Vd, Vs, Vt);    \
    }                               \
    if (sizeof(T) == 4) {           \
      if (std::is_signed<T>::value) \
        vacc_w_vv(Vd, Vs, Vt);      \
      else                          \
        vacc_w_u_vv(Vd, Vs, Vt);    \
    }                               \
  }

#define vacc_v_m(T, Vd, Vs, Vt)     \
  {                                 \
    if (sizeof(T) == 2) {           \
      if (std::is_signed<T>::value) \
        vacc_h_vv_m(Vd, Vs, Vt);    \
      else                          \
        vacc_h_u_vv_m(Vd, Vs, Vt);  \
    }                               \
    if (sizeof(T) == 4) {           \
      if (std::is_signed<T>::value) \
        vacc_w_vv_m(Vd, Vs, Vt);    \
      else                          \
        vacc_w_u_vv_m(Vd, Vs, Vt);  \
    }                               \
  }

template <typename T1, typename T2>
static void test_vacc() {
  static_assert(sizeof(T1) == 2 * sizeof(T2));
  static_assert(std::is_signed<T1>::value == std::is_signed<T2>::value);

  constexpr int n = sizeof(T1);
  int lanes;

  getmaxvl(T1, lanes);

  T1 acc[2][lanes] __attribute__((aligned(64)));
  T1 dut[2][lanes] __attribute__((aligned(64)));
  T1 ref[2][lanes] __attribute__((aligned(64)));
  T2 inp[lanes * 2] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    acc[0][i] = (0x40 << (8 * (n - 1)));
    acc[1][i] = (0x80 << (8 * (n - 1)));
  }
  for (int j = 0; j < lanes * 2; ++j) {
    inp[j] = -lanes + j;
  }
  for (int i = 0; i < lanes; ++i) {
    ref[0][i] = acc[0][i] + inp[2 * i + 0];
    ref[1][i] = acc[1][i] + inp[2 * i + 1];
  }

  vld_x(T1, v0, acc[0]);
  vld_x(T1, v1, acc[1]);
  vld_x(T2, v8, inp);
  vacc_v(T1, v4, v0, v8);
  vst_x(T1, v4, dut[0]);
  vst_x(T1, v5, dut[1]);

  for (int i = 0; i < lanes; ++i) {
    if (ref[0][i] != dut[0][i] || ref[1][i] != dut[1][i]) {
      printf("**error vacc[%d]\n", i);
      exit(-1);
    }
  }
}

template <typename T1, typename T2>
static void test_vacc_m() {
  static_assert(sizeof(T1) == 2 * sizeof(T2));
  static_assert(std::is_signed<T1>::value == std::is_signed<T2>::value);

  constexpr int n = sizeof(T1);
  int lanes;

  getmaxvl_m(T1, lanes);

  T1 acc[2][lanes] __attribute__((aligned(64)));
  T1 dut[2][lanes] __attribute__((aligned(64)));
  T1 ref[2][lanes] __attribute__((aligned(64)));
  T2 inp[lanes * 2] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    acc[0][i] = (0x40 << (8 * (n - 1)));
    acc[1][i] = (0x80 << (8 * (n - 1)));
  }
  for (int j = 0; j < lanes * 2; ++j) {
    inp[j] = -lanes + j;
  }
  for (int i = 0; i < lanes; ++i) {
    ref[0][i] = acc[0][i] + inp[2 * i + 0];
    ref[1][i] = acc[1][i] + inp[2 * i + 1];
  }

  vld_x_m(T1, v0, acc[0]);
  vld_x_m(T1, v4, acc[1]);
  vld_x_m(T2, v16, inp);
  vacc_v_m(T1, v8, v0, v16);
  vst_x_m(T1, v8, dut[0]);
  vst_x_m(T1, v12, dut[1]);

  for (int i = 0; i < lanes; ++i) {
    if (ref[0][i] != dut[0][i] || ref[1][i] != dut[1][i]) {
      printf("**error vacc_m[%d]\n", i);
      // printf("%x %x : %x %x\n", ref[0][i], dut[0][i], ref[1][i], dut[1][i]);
      exit(-1);
    }
  }
}

int main() {
  test_vacc<int16_t, int8_t>();
  test_vacc<uint16_t, uint8_t>();
  test_vacc<int32_t, int16_t>();
  test_vacc<uint32_t, uint16_t>();

  test_vacc_m<int16_t, int8_t>();
  test_vacc_m<uint16_t, uint8_t>();
  test_vacc_m<int32_t, int16_t>();
  test_vacc_m<uint32_t, uint16_t>();

  return 0;
}
