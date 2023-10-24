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

#define vmvp_vx(T, Vd, Vs, t) \
  {                           \
    if (sizeof(T) == 1) {     \
      vmvp_b_vx(Vd, Vs, t);   \
    }                         \
    if (sizeof(T) == 2) {     \
      vmvp_h_vx(Vd, Vs, t);   \
    }                         \
    if (sizeof(T) == 4) {     \
      vmvp_w_vx(Vd, Vs, t);   \
    }                         \
  }

#define vmvp_vx_m(T, Vd, Vs, t) \
  {                             \
    if (sizeof(T) == 1) {       \
      vmvp_b_vx_m(Vd, Vs, t);   \
    }                           \
    if (sizeof(T) == 2) {       \
      vmvp_h_vx_m(Vd, Vs, t);   \
    }                           \
    if (sizeof(T) == 4) {       \
      vmvp_w_vx_m(Vd, Vs, t);   \
    }                           \
  }

template <int m>
void test_vmv_v() {
  int vlenb;
  if (m)
    getmaxvl_b_m(vlenb);
  else
    getmaxvl_b(vlenb);

  uint8_t ref[vlenb] __attribute__((aligned(64)));
  uint8_t dut[vlenb] __attribute__((aligned(64)));

  for (int i = 0; i < vlenb; ++i) {
    ref[i] = i;
  }

  if (m) {
    vld_b_x_m(v8, ref);
    vmv_v_m(v0, v8);
    vst_b_x_m(v0, dut);
  } else {
    vld_b_x(v3, ref);
    vmv_v(v0, v3);
    vst_b_x(v0, dut);
  }

  for (int i = 0; i < vlenb; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vmv_v[%d][%d] %02x %02x\n", m, i, ref[i], dut[i]);
      exit(-1);
    }
  }
}

template <int m>
void test_vmvp_vv() {
  int vlenb;
  if (m)
    getmaxvl_b_m(vlenb);
  else
    getmaxvl_b(vlenb);

  uint8_t ref[2][vlenb] __attribute__((aligned(64)));
  uint8_t dut[2][vlenb] __attribute__((aligned(64)));

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < vlenb; ++i) {
      ref[j][i] = i + (j << 7);
    }
  }

  if (m) {
    vld_b_x_m(v16, ref[0]);
    vld_b_x_m(v32, ref[1]);
    vmvp_vv_m(v0, v16, v32);
    vst_b_x_m(v0, dut[0]);
    vst_b_x_m(v4, dut[1]);
  } else {
    vld_b_x(v3, ref[0]);
    vld_b_x(v5, ref[1]);
    vmvp_vv(v0, v3, v5);
    vst_b_x(v0, dut[0]);
    vst_b_x(v1, dut[1]);
  }

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < vlenb; ++i) {
      if (ref[j][i] != dut[j][i]) {
        printf("**error vmvp_vv[%d][%d,%d] %02x %02x\n", m, j, i, ref[j][i],
               dut[j][i]);
        exit(-1);
      }
    }
  }
}

template <int m, typename T>
void test_vmvp_vx() {
  int vlenb;
  if (m) {
    getmaxvl_m(T, vlenb);
  } else {
    getmaxvl(T, vlenb);
  }

  T ref[2][vlenb] __attribute__((aligned(64)));
  T dut[2][vlenb] __attribute__((aligned(64)));
  T scalar = T(0x12345678);

  for (int i = 0; i < vlenb; ++i) {
    ref[0][i] = i;
    ref[1][i] = scalar;
  }

  if (m) {
    vld_b_x_m(v16, ref[0]);
    vmvp_vx_m(T, v0, v16, scalar);
    vst_b_x_m(v0, dut[0]);
    vst_b_x_m(v4, dut[1]);
  } else {
    vld_b_x(v3, ref[0]);
    vmvp_vx(T, v0, v3, scalar);
    vst_b_x(v0, dut[0]);
    vst_b_x(v1, dut[1]);
  }

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < vlenb; ++i) {
      if (ref[j][i] != dut[j][i]) {
        printf("**error vmvp_vx[%d][%d,%d] ", m, j, i);
        printf(PrintfTraits<T>::kFmtHex, ref[j][i]);
        printf(" ");
        printf(PrintfTraits<T>::kFmtHex, dut[j][i]);
        printf("\n");
      }
    }
  }
}

int main() {
  test_vmv_v<0>();
  test_vmv_v<1>();

  test_vmvp_vv<0>();
  test_vmvp_vv<1>();

  test_vmvp_vx<0, uint8_t>();
  test_vmvp_vx<0, uint16_t>();
  test_vmvp_vx<0, uint32_t>();
  test_vmvp_vx<1, uint8_t>();
  test_vmvp_vx<1, uint16_t>();
  test_vmvp_vx<1, uint32_t>();

  return 0;
}
