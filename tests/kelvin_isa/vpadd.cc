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

#include "crt/printf_traits.h"
#include "tests/kelvin_isa/kelvin_test.h"

#define vpadd_v(T, Vd, Vs)          \
  {                                 \
    if (sizeof(T) == 2) {           \
      if (std::is_signed<T>::value) \
        vpadd_h_v(Vd, Vs);          \
      else                          \
        vpadd_h_u_v(Vd, Vs);        \
    }                               \
    if (sizeof(T) == 4) {           \
      if (std::is_signed<T>::value) \
        vpadd_w_v(Vd, Vs);          \
      else                          \
        vpadd_w_u_v(Vd, Vs);        \
    }                               \
  }

#define vpadd_v_m(T, Vd, Vs)        \
  {                                 \
    if (sizeof(T) == 2) {           \
      if (std::is_signed<T>::value) \
        vpadd_h_v_m(Vd, Vs);        \
      else                          \
        vpadd_h_u_v_m(Vd, Vs);      \
    }                               \
    if (sizeof(T) == 4) {           \
      if (std::is_signed<T>::value) \
        vpadd_w_v_m(Vd, Vs);        \
      else                          \
        vpadd_w_u_v_m(Vd, Vs);      \
    }                               \
  }

template <typename T1, typename T2>
static void test_vpadd_v() {
  constexpr int n = sizeof(T1);
  int lanes;
  getmaxvl(T2, lanes);

  T1 inp[lanes * 2] __attribute__((aligned(64)));
  T2 ref[lanes] __attribute__((aligned(64)));
  T2 dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes * 2; ++i) {
    inp[i] = ((i & 1 ? 0xc0 : 0x40) << (8 * (n - 1))) + i - 4;
  }

  for (int i = 0; i < lanes; ++i) {
    ref[i] = T2(inp[2 * i + 0]) + T2(inp[2 * i + 1]);
  }

  vld_x(T1, v16, inp);
  vpadd_v(T2, v0, v16);
  vst_x(T2, v0, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vpadd_v[%d] ", i);
      printf(PrintfTraits<T2>::kFmt, ref[i]);
      printf(" ");
      printf(PrintfTraits<T2>::kFmt, dut[i]);
      printf("\n");
      printf("  inputs: ");
      printf(PrintfTraits<T1>::kFmt, inp[2 * i + 0]);
      printf(", ");
      printf(PrintfTraits<T1>::kFmt, inp[2 * i + 1]);
      printf("\n");
      exit(-1);
    }
  }
}

template <typename T1, typename T2>
static void test_vpadd_v_m() {
  constexpr int n = sizeof(T1);
  int lanes;
  getmaxvl_m(T2, lanes);

  T1 inp[lanes * 2] __attribute__((aligned(64)));
  T2 ref[lanes] __attribute__((aligned(64)));
  T2 dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes * 2; ++i) {
    inp[i] = ((i & 1 ? 0xc0 : 0x40) << (8 * (n - 1))) + i - 4;
  }

  for (int i = 0; i < lanes; ++i) {
    ref[i] = T2(inp[2 * i + 0]) + T2(inp[2 * i + 1]);
  }

  vld_x_m(T1, v16, inp);
  vpadd_v_m(T2, v0, v16);
  vst_x_m(T2, v0, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vpadd_v_m[%d] ", i);
      printf(PrintfTraits<T2>::kFmtHex, ref[i]);
      printf(" ");
      printf(PrintfTraits<T2>::kFmtHex, dut[i]);
      printf("\n");
      exit(-1);
    }
  }
}

int main() {
  test_vpadd_v<int8_t, int16_t>();
  test_vpadd_v<int16_t, int32_t>();
  test_vpadd_v<uint8_t, uint16_t>();
  test_vpadd_v<uint16_t, uint32_t>();

  test_vpadd_v_m<int8_t, int16_t>();
  test_vpadd_v_m<int16_t, int32_t>();
  test_vpadd_v_m<uint8_t, uint16_t>();
  test_vpadd_v_m<uint16_t, uint32_t>();

  return 0;
}
