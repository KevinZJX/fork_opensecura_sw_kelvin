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

#define vsliden_1_vv(T, Vd, Vs, Vt) \
  {                                 \
    if (sizeof(T) == 1) {           \
      vsliden_b_1_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 2) {           \
      vsliden_h_1_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 4) {           \
      vsliden_w_1_vv(Vd, Vs, Vt);   \
    }                               \
  }
#define vsliden_2_vv(T, Vd, Vs, Vt) \
  {                                 \
    if (sizeof(T) == 1) {           \
      vsliden_b_2_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 2) {           \
      vsliden_h_2_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 4) {           \
      vsliden_w_2_vv(Vd, Vs, Vt);   \
    }                               \
  }
#define vsliden_3_vv(T, Vd, Vs, Vt) \
  {                                 \
    if (sizeof(T) == 1) {           \
      vsliden_b_3_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 2) {           \
      vsliden_h_3_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 4) {           \
      vsliden_w_3_vv(Vd, Vs, Vt);   \
    }                               \
  }
#define vsliden_4_vv(T, Vd, Vs, Vt) \
  {                                 \
    if (sizeof(T) == 1) {           \
      vsliden_b_4_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 2) {           \
      vsliden_h_4_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 4) {           \
      vsliden_w_4_vv(Vd, Vs, Vt);   \
    }                               \
  }
#define vslidep_1_vv(T, Vd, Vs, Vt) \
  {                                 \
    if (sizeof(T) == 1) {           \
      vslidep_b_1_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 2) {           \
      vslidep_h_1_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 4) {           \
      vslidep_w_1_vv(Vd, Vs, Vt);   \
    }                               \
  }
#define vslidep_2_vv(T, Vd, Vs, Vt) \
  {                                 \
    if (sizeof(T) == 1) {           \
      vslidep_b_2_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 2) {           \
      vslidep_h_2_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 4) {           \
      vslidep_w_2_vv(Vd, Vs, Vt);   \
    }                               \
  }
#define vslidep_3_vv(T, Vd, Vs, Vt) \
  {                                 \
    if (sizeof(T) == 1) {           \
      vslidep_b_3_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 2) {           \
      vslidep_h_3_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 4) {           \
      vslidep_w_3_vv(Vd, Vs, Vt);   \
    }                               \
  }
#define vslidep_4_vv(T, Vd, Vs, Vt) \
  {                                 \
    if (sizeof(T) == 1) {           \
      vslidep_b_4_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 2) {           \
      vslidep_h_4_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 4) {           \
      vslidep_w_4_vv(Vd, Vs, Vt);   \
    }                               \
  }

#define vslidehn_1_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidehn_b_1_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidehn_h_1_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidehn_w_1_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidehn_2_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidehn_b_2_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidehn_h_2_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidehn_w_2_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidehn_3_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidehn_b_3_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidehn_h_3_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidehn_w_3_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidehn_4_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidehn_b_4_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidehn_h_4_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidehn_w_4_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidehp_1_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidehp_b_1_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidehp_h_1_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidehp_w_1_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidehp_2_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidehp_b_2_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidehp_h_2_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidehp_w_2_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidehp_3_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidehp_b_3_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidehp_h_3_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidehp_w_3_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidehp_4_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidehp_b_4_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidehp_h_4_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidehp_w_4_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }

#define vslidevn_1_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidevn_b_1_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidevn_h_1_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidevn_w_1_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidevn_2_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidevn_b_2_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidevn_h_2_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidevn_w_2_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidevn_3_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidevn_b_3_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidevn_h_3_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidevn_w_3_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidevn_4_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidevn_b_4_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidevn_h_4_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidevn_w_4_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidevp_1_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidevp_b_1_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidevp_h_1_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidevp_w_1_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidevp_2_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidevp_b_2_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidevp_h_2_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidevp_w_2_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidevp_3_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidevp_b_3_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidevp_h_3_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidevp_w_3_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }
#define vslidevp_4_vv_m(T, Vd, Vs, Vt) \
  {                                    \
    if (sizeof(T) == 1) {              \
      vslidevp_b_4_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 2) {              \
      vslidevp_h_4_vv_m(Vd, Vs, Vt);   \
    }                                  \
    if (sizeof(T) == 4) {              \
      vslidevp_w_4_vv_m(Vd, Vs, Vt);   \
    }                                  \
  }

template <int s, typename T>
void test_vsliden() {
  int lanes;
  getmaxvl(T, lanes);

  T in0[lanes] __attribute__((aligned(64)));
  T in1[lanes] __attribute__((aligned(64)));
  T ref[lanes] __attribute__((aligned(64)));
  T dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    in0[i] = krand();
    in1[i] = krand();
  }

  for (int i = 0; i < lanes; ++i) {
    ref[i] = i < (lanes - s) ? in0[i + s] : in1[i - (lanes - s)];
  }

  vld_x(T, v0, in0);
  vld_x(T, v1, in1);
  if (s == 1) vsliden_1_vv(T, v2, v0, v1);
  if (s == 2) vsliden_2_vv(T, v2, v0, v1);
  if (s == 3) vsliden_3_vv(T, v2, v0, v1);
  if (s == 4) vsliden_4_vv(T, v2, v0, v1);
  vst_x(T, v2, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vsliden<%d>[%d] ", s, i);
      printf(PrintfTraits<T>::kFmtHex, ref[i]);
      printf(" ");
      printf(PrintfTraits<T>::kFmtHex, dut[i]);
      printf("\n");
      exit(-1);
    }
  }
}

template <int s, typename T>
void test_vslidep() {
  int lanes;
  getmaxvl(T, lanes);

  T in0[lanes] __attribute__((aligned(64)));
  T in1[lanes] __attribute__((aligned(64)));
  T ref[lanes] __attribute__((aligned(64)));
  T dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    in0[i] = krand();
    in1[i] = krand();
  }

  for (int i = 0; i < lanes; ++i) {
    ref[i] = i - s < 0 ? in0[lanes + (i - s)] : in1[i - s];
  }

  vld_x(T, v0, in0);
  vld_x(T, v1, in1);
  if (s == 1) vslidep_1_vv(T, v2, v0, v1);
  if (s == 2) vslidep_2_vv(T, v2, v0, v1);
  if (s == 3) vslidep_3_vv(T, v2, v0, v1);
  if (s == 4) vslidep_4_vv(T, v2, v0, v1);
  vst_x(T, v2, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vslidep<%d>[%d] ", s, i);
      printf(PrintfTraits<T>::kFmtHex, ref[i]);
      printf(" ");
      printf(PrintfTraits<T>::kFmtHex, dut[i]);
      printf("\n");
      exit(-1);
    }
  }
}

template <int s, typename T>
void test_vslidehn_m() {
  int lanes;
  getmaxvl_m(T, lanes);

  T in0[lanes] __attribute__((aligned(64)));
  T in1[lanes] __attribute__((aligned(64)));
  T ref[lanes] __attribute__((aligned(64)));
  T dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    in0[i] = krand();
    in1[i] = krand();
  }

  for (int i = 0; i < lanes; ++i) {
    ref[i] = i < (lanes - s) ? in0[i + s] : in1[i - (lanes - s)];
  }

  vld_x_m(T, v0, in0);
  vld_x_m(T, v4, in1);
  if (s == 1) vslidehn_1_vv_m(T, v8, v0, v4);
  if (s == 2) vslidehn_2_vv_m(T, v8, v0, v4);
  if (s == 3) vslidehn_3_vv_m(T, v8, v0, v4);
  if (s == 4) vslidehn_4_vv_m(T, v8, v0, v4);
  vst_x_m(T, v8, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vslidehn_m<%d>[%d] ", s, i);
      printf(PrintfTraits<T>::kFmtHex, ref[i]);
      printf(" ");
      printf(PrintfTraits<T>::kFmtHex, dut[i]);
      printf("\n");
      exit(-1);
    }
  }
}

template <int s, typename T>
void test_vslidehp_m() {
  int lanes;
  getmaxvl_m(T, lanes);

  T in0[lanes] __attribute__((aligned(64)));
  T in1[lanes] __attribute__((aligned(64)));
  T ref[lanes] __attribute__((aligned(64)));
  T dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    in0[i] = krand();
    in1[i] = krand();
  }

  for (int i = 0; i < lanes; ++i) {
    ref[i] = i - s < 0 ? in0[lanes + (i - s)] : in1[i - s];
  }

  vld_x_m(T, v0, in0);
  vld_x_m(T, v4, in1);
  if (s == 1) vslidehp_1_vv_m(T, v8, v0, v4);
  if (s == 2) vslidehp_2_vv_m(T, v8, v0, v4);
  if (s == 3) vslidehp_3_vv_m(T, v8, v0, v4);
  if (s == 4) vslidehp_4_vv_m(T, v8, v0, v4);
  vst_x_m(T, v8, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vslidehp_m<%d>[%d] ", s, i);
      printf(PrintfTraits<T>::kFmtHex, ref[i]);
      printf(" ");
      printf(PrintfTraits<T>::kFmtHex, dut[i]);
      printf("\n");
      exit(-1);
    }
  }
}

template <int s, typename T>
void test_vslidevn_m() {
  int lanes;
  getmaxvl_m(T, lanes);

  T in0[lanes] __attribute__((aligned(64)));
  T in1[lanes] __attribute__((aligned(64)));
  T ref[lanes] __attribute__((aligned(64)));
  T dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    in0[i] = krand();
    in1[i] = krand();
  }

  int k = lanes / 4;
  for (int m = 0; m < 4; ++m) {
    for (int i = 0; i < k; ++i) {
      ref[m * k + i] =
          i < (k - s) ? in0[m * k + i + s] : in1[m * k + i - (k - s)];
    }
  }

  vld_x_m(T, v0, in0);
  vld_x_m(T, v4, in1);
  if (s == 1) vslidevn_1_vv_m(T, v8, v0, v4);
  if (s == 2) vslidevn_2_vv_m(T, v8, v0, v4);
  if (s == 3) vslidevn_3_vv_m(T, v8, v0, v4);
  if (s == 4) vslidevn_4_vv_m(T, v8, v0, v4);
  vst_x_m(T, v8, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vslidevn_m<%d>[%d] ", s, i);
      printf(PrintfTraits<T>::kFmtHex, ref[i]);
      printf(" ");
      printf(PrintfTraits<T>::kFmtHex, dut[i]);
      printf("\n");
      exit(-1);
    }
  }
}

template <int s, typename T>
void test_vslidevp_m() {
  int lanes;
  getmaxvl_m(T, lanes);

  T in0[lanes] __attribute__((aligned(64)));
  T in1[lanes] __attribute__((aligned(64)));
  T ref[lanes] __attribute__((aligned(64)));
  T dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes; ++i) {
    in0[i] = krand();
    in1[i] = krand();
  }

  int k = lanes / 4;
  for (int m = 0; m < 4; ++m) {
    for (int i = 0; i < k; ++i) {
      ref[m * k + i] =
          i - s < 0 ? in0[k + (m * k + i - s)] : in1[m * k + i - s];
    }
  }

  vld_x_m(T, v0, in0);
  vld_x_m(T, v4, in1);
  if (s == 1) vslidevp_1_vv_m(T, v8, v0, v4);
  if (s == 2) vslidevp_2_vv_m(T, v8, v0, v4);
  if (s == 3) vslidevp_3_vv_m(T, v8, v0, v4);
  if (s == 4) vslidevp_4_vv_m(T, v8, v0, v4);
  vst_x_m(T, v8, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vslidevp_m<%d>[%d] ", s, i);
      printf(PrintfTraits<T>::kFmtHex, ref[i]);
      printf(" ");
      printf(PrintfTraits<T>::kFmtHex, dut[i]);
      printf("\n");
      exit(-1);
    }
  }
}

int main() {
  test_vsliden<1, uint8_t>();
  test_vsliden<2, uint8_t>();
  test_vsliden<3, uint8_t>();
  test_vsliden<4, uint8_t>();

  test_vsliden<1, uint16_t>();
  test_vsliden<2, uint16_t>();
  test_vsliden<3, uint16_t>();
  test_vsliden<4, uint16_t>();

  test_vsliden<1, uint32_t>();
  test_vsliden<2, uint32_t>();
  test_vsliden<3, uint32_t>();
  test_vsliden<4, uint32_t>();

  test_vslidep<1, uint8_t>();
  test_vslidep<2, uint8_t>();
  test_vslidep<3, uint8_t>();
  test_vslidep<4, uint8_t>();

  test_vslidep<1, uint16_t>();
  test_vslidep<2, uint16_t>();
  test_vslidep<3, uint16_t>();
  test_vslidep<4, uint16_t>();

  test_vslidep<1, uint32_t>();
  test_vslidep<2, uint32_t>();
  test_vslidep<3, uint32_t>();
  test_vslidep<4, uint32_t>();

  test_vslidevn_m<1, uint8_t>();
  test_vslidevn_m<2, uint8_t>();
  test_vslidevn_m<3, uint8_t>();
  test_vslidevn_m<4, uint8_t>();

  test_vslidevn_m<1, uint16_t>();
  test_vslidevn_m<2, uint16_t>();
  test_vslidevn_m<3, uint16_t>();
  test_vslidevn_m<4, uint16_t>();

  test_vslidevn_m<1, uint32_t>();
  test_vslidevn_m<2, uint32_t>();
  test_vslidevn_m<3, uint32_t>();
  test_vslidevn_m<4, uint32_t>();

  test_vslidevp_m<1, uint8_t>();
  test_vslidevp_m<2, uint8_t>();
  test_vslidevp_m<3, uint8_t>();
  test_vslidevp_m<4, uint8_t>();

  test_vslidevp_m<1, uint16_t>();
  test_vslidevp_m<2, uint16_t>();
  test_vslidevp_m<3, uint16_t>();
  test_vslidevp_m<4, uint16_t>();

  test_vslidevp_m<1, uint32_t>();
  test_vslidevp_m<2, uint32_t>();
  test_vslidevp_m<3, uint32_t>();
  test_vslidevp_m<4, uint32_t>();

  test_vslidehn_m<1, uint8_t>();
  test_vslidehn_m<2, uint8_t>();
  test_vslidehn_m<3, uint8_t>();
  test_vslidehn_m<4, uint8_t>();

  test_vslidehn_m<1, uint16_t>();
  test_vslidehn_m<2, uint16_t>();
  test_vslidehn_m<3, uint16_t>();
  test_vslidehn_m<4, uint16_t>();

  test_vslidehn_m<1, uint32_t>();
  test_vslidehn_m<2, uint32_t>();
  test_vslidehn_m<3, uint32_t>();
  test_vslidehn_m<4, uint32_t>();

  test_vslidehp_m<1, uint8_t>();
  test_vslidehp_m<2, uint8_t>();
  test_vslidehp_m<3, uint8_t>();
  test_vslidehp_m<4, uint8_t>();

  test_vslidehp_m<1, uint16_t>();
  test_vslidehp_m<2, uint16_t>();
  test_vslidehp_m<3, uint16_t>();
  test_vslidehp_m<4, uint16_t>();

  test_vslidehp_m<1, uint32_t>();
  test_vslidehp_m<2, uint32_t>();
  test_vslidehp_m<3, uint32_t>();
  test_vslidehp_m<4, uint32_t>();

  return 0;
}
