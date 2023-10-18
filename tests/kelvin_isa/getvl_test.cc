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

// clang-format off
#define TEST_GETVL_X(T, op, in)                                           \
  {                                                                       \
    int ref, dut;                                                         \
    if (sizeof(T) == 1) {                                                 \
      ref = std::min(vlb, in);                                            \
    } else if (sizeof(T) == 2) {                                          \
      ref = std::min(vlh, in);                                            \
    } else if (sizeof(T) == 4) {                                          \
      ref = std::min(vlw, in);                                            \
    } else {                                                              \
      printf("**error(%d)[unknown getvl]\n", __LINE__);                   \
      exit(-1);                                                           \
    }                                                                     \
    __asm__ __volatile_always__(ARGS_F_A_A(op, %0, %1)                    \
                                : "=r"(dut)                               \
                                : "r"(in));                               \
    if (ref != dut) {                                                     \
      printf("**error(%d)[%s] %d : %d %d\n", __LINE__, op, in, ref, dut); \
      exit(-1);                                                           \
    }                                                                     \
  }
#define TEST_GETVL_XX(T, op, in0, in1)                                       \
  {                                                                          \
    int ref, dut;                                                            \
    if (sizeof(T) == 1) {                                                    \
      ref = std::min(vlb, std::min(in0, in1));                               \
    } else if (sizeof(T) == 2) {                                             \
      ref = std::min(vlh, std::min(in0, in1));                               \
    } else if (sizeof(T) == 4) {                                             \
      ref = std::min(vlw, std::min(in0, in1));                               \
    } else {                                                                 \
      printf("**error(%d)[unknown getvl]\n", __LINE__);                      \
      exit(-1);                                                              \
    }                                                                        \
    __asm__ __volatile_always__(ARGS_F_A_A_A(op, %0, %1, %2)                 \
                                : "=r"(dut)                                  \
                                : "r"(in0), "r"(in1));                       \
    if (ref != dut) {                                                        \
      printf("**error(%d)[%s] %d %d : %d %d\n", __LINE__, op, in0, in1, ref, \
             dut);                                                           \
      exit(-1);                                                              \
    }                                                                        \
  }
// clang-format on

int main() {
  const int pad = 3;
  int vlb, vlh, vlw;
  // ---------------------------------------------------------------------------
  // Test baseline.
  getmaxvl_w(vlw);
  getmaxvl_h(vlh);
  getmaxvl_b(vlb);
  if (vlw != VLENW) {
    printf("**error(%d)[%s] %d\n", __LINE__, "getmaxvl.w", vlw);
    exit(-1);
  }
  if (vlh != vlw * 2) {
    printf("**error(%d)[%s] %d\n", __LINE__, "getmaxvl.h", vlh);
    exit(-1);
  }
  if (vlb != vlw * 4) {
    printf("**error(%d)[%s] %d\n", __LINE__, "getmaxvl.b", vlb);
    exit(-1);
  }
  for (int i = 0; i < vlb + pad; ++i) {
    TEST_GETVL_X(uint8_t, "getvl.b.x", i);
  }
  for (int i = 0; i < vlh + pad; ++i) {
    TEST_GETVL_X(uint16_t, "getvl.h.x", i);
  }
  for (int i = 0; i < vlw + pad; ++i) {
    TEST_GETVL_X(uint32_t, "getvl.w.x", i);
  }
  for (int i = 0; i < vlb + pad; ++i) {
    for (int j = 0; j < vlb + pad; ++j) {
      TEST_GETVL_XX(uint8_t, "getvl.b.xx", i, j);
    }
  }
  for (int i = 0; i < vlh + pad; ++i) {
    for (int j = 0; j < vlh + pad; ++j) {
      TEST_GETVL_XX(uint16_t, "getvl.h.xx", i, j);
    }
  }
  for (int i = 0; i < vlw + pad; ++i) {
    for (int j = 0; j < vlw + pad; ++j) {
      TEST_GETVL_XX(uint32_t, "getvl.w.xx", i, j);
    }
  }
  // ---------------------------------------------------------------------------
  // Test stripmine.
  int vlw_p = vlw;
  getmaxvl_w_m(vlw);
  getmaxvl_h_m(vlh);
  getmaxvl_b_m(vlb);
  if (vlw != 4 * vlw_p) {
    printf("**error(%d)[%s] %d\n", __LINE__, "getmaxvl.w.m", vlw);
    exit(-1);
  }
  if (vlh != vlw * 2) {
    printf("**error(%d)[%s] %d\n", __LINE__, "getmaxvl.h.m", vlh);
    exit(-1);
  }
  if (vlb != vlw * 4) {
    printf("**error(%d)[%s] %d\n", __LINE__, "getmaxvl.b.m", vlb);
    exit(-1);
  }
  for (int i = 0; i < vlb + pad; ++i) {
    TEST_GETVL_X(uint8_t, "getvl.b.x.m", i);
  }
  for (int i = 0; i < vlh + pad; ++i) {
    TEST_GETVL_X(uint16_t, "getvl.h.x.m", i);
  }
  for (int i = 0; i < vlw + pad; ++i) {
    TEST_GETVL_X(uint32_t, "getvl.w.x.m", i);
  }
  for (int i = 0; i < vlb + pad; ++i) {
    for (int j = 0; j < vlb + pad; ++j) {
      TEST_GETVL_XX(uint8_t, "getvl.b.xx.m", i, j);
    }
  }
  for (int i = 0; i < vlh + pad; ++i) {
    for (int j = 0; j < vlh + pad; ++j) {
      TEST_GETVL_XX(uint16_t, "getvl.h.xx.m", i, j);
    }
  }
  for (int i = 0; i < vlw + pad; ++i) {
    for (int j = 0; j < vlw + pad; ++j) {
      TEST_GETVL_XX(uint32_t, "getvl.w.xx.m", i, j);
    }
  }
  return 0;
}
