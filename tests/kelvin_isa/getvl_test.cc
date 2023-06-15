// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//

#include "tests/kelvin_isa/kelvin_test.h"

// clang-format off
#define TEST_GETVL_X(op, in)                                              \
  {                                                                       \
    int ref, dut;                                                         \
    if (op == "getvl.b.x" || op == "getvl.b.x.m") {                       \
      ref = std::min(vlb, in);                                            \
    } else if (op == "getvl.h.x" || op == "getvl.h.x.m") {                \
      ref = std::min(vlh, in);                                            \
    } else if (op == "getvl.w.x" || op == "getvl.w.x.m") {                \
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
#define TEST_GETVL_XX(op, in0, in1)                                          \
  {                                                                          \
    int ref, dut;                                                            \
    if (op == "getvl.b.xx" || op == "getvl.b.xx.m") {                        \
      ref = std::min(vlb, std::min(in0, in1));                               \
    } else if (op == "getvl.h.xx" || op == "getvl.h.xx.m") {                 \
      ref = std::min(vlh, std::min(in0, in1));                               \
    } else if (op == "getvl.w.xx" || op == "getvl.w.xx.m") {                 \
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
    TEST_GETVL_X("getvl.b.x", i);
  }
  for (int i = 0; i < vlh + pad; ++i) {
    TEST_GETVL_X("getvl.h.x", i);
  }
  for (int i = 0; i < vlw + pad; ++i) {
    TEST_GETVL_X("getvl.w.x", i);
  }
  for (int i = 0; i < vlb + pad; ++i) {
    for (int j = 0; j < vlb + pad; ++j) {
      TEST_GETVL_XX("getvl.b.xx", i, j);
    }
  }
  for (int i = 0; i < vlh + pad; ++i) {
    for (int j = 0; j < vlh + pad; ++j) {
      TEST_GETVL_XX("getvl.h.xx", i, j);
    }
  }
  for (int i = 0; i < vlw + pad; ++i) {
    for (int j = 0; j < vlw + pad; ++j) {
      TEST_GETVL_XX("getvl.w.xx", i, j);
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
    TEST_GETVL_X("getvl.b.x.m", i);
  }
  for (int i = 0; i < vlh + pad; ++i) {
    TEST_GETVL_X("getvl.h.x.m", i);
  }
  for (int i = 0; i < vlw + pad; ++i) {
    TEST_GETVL_X("getvl.w.x.m", i);
  }
  for (int i = 0; i < vlb + pad; ++i) {
    for (int j = 0; j < vlb + pad; ++j) {
      TEST_GETVL_XX("getvl.b.xx.m", i, j);
    }
  }
  for (int i = 0; i < vlh + pad; ++i) {
    for (int j = 0; j < vlh + pad; ++j) {
      TEST_GETVL_XX("getvl.h.xx.m", i, j);
    }
  }
  for (int i = 0; i < vlw + pad; ++i) {
    for (int j = 0; j < vlw + pad; ++j) {
      TEST_GETVL_XX("getvl.w.xx.m", i, j);
    }
  }
  return 0;
}
