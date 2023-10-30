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

// Kelvin ISA test common header

#ifndef TESTS_KELVIN_ISA_KELVIN_TEST_H_
#define TESTS_KELVIN_ISA_KELVIN_TEST_H_

#include <cstdio>

#include "crt/kelvin.h"

// Maximum storage required for parameterized machine load/store.
constexpr int VLEN = 256;  // simd register bits. Need to match the HW parameter
constexpr int VLENB = VLEN / 8;
constexpr int VLENH = VLEN / 16;
constexpr int VLENW = VLEN / 32;

uint32_t krand(void) {
  static uint32_t x = 123456789;
  static uint32_t y = 362436069;
  static uint32_t z = 521288629;
  static uint32_t w = 88675123;
  uint32_t t = x ^ (x << 11);
  x = y;
  y = z;
  z = w;
  return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}

#define getmaxvl(T, d)                 \
  {                                    \
    if (sizeof(T) == 1) getmaxvl_b(d); \
    if (sizeof(T) == 2) getmaxvl_h(d); \
    if (sizeof(T) == 4) getmaxvl_w(d); \
  }

#define getmaxvl_m(T, d)                 \
  {                                      \
    if (sizeof(T) == 1) getmaxvl_b_m(d); \
    if (sizeof(T) == 2) getmaxvl_h_m(d); \
    if (sizeof(T) == 4) getmaxvl_w_m(d); \
  }

#define vld_x(T, Vd, s)                 \
  {                                     \
    if (sizeof(T) == 1) vld_b_x(Vd, s); \
    if (sizeof(T) == 2) vld_h_x(Vd, s); \
    if (sizeof(T) == 4) vld_w_x(Vd, s); \
  }

#define vst_x(T, Vd, s)                 \
  {                                     \
    if (sizeof(T) == 1) vst_b_x(Vd, s); \
    if (sizeof(T) == 2) vst_h_x(Vd, s); \
    if (sizeof(T) == 4) vst_w_x(Vd, s); \
  }

#define vld_x_m(T, Vd, s)                 \
  {                                       \
    if (sizeof(T) == 1) vld_b_x_m(Vd, s); \
    if (sizeof(T) == 2) vld_h_x_m(Vd, s); \
    if (sizeof(T) == 4) vld_w_x_m(Vd, s); \
  }

#define vst_x_m(T, Vd, s)                 \
  {                                       \
    if (sizeof(T) == 1) vst_b_x_m(Vd, s); \
    if (sizeof(T) == 2) vst_h_x_m(Vd, s); \
    if (sizeof(T) == 4) vst_w_x_m(Vd, s); \
  }

#define vld_l_xx(T, Vd, s, t)                 \
  {                                           \
    if (sizeof(T) == 1) vld_b_l_xx(Vd, s, t); \
    if (sizeof(T) == 2) vld_h_l_xx(Vd, s, t); \
    if (sizeof(T) == 4) vld_w_l_xx(Vd, s, t); \
  }

#define vst_l_xx(T, Vd, s, t)                 \
  {                                           \
    if (sizeof(T) == 1) vst_b_l_xx(Vd, s, t); \
    if (sizeof(T) == 2) vst_h_l_xx(Vd, s, t); \
    if (sizeof(T) == 4) vst_w_l_xx(Vd, s, t); \
  }

#define vld_l_xx_m(T, Vd, s, t)                 \
  {                                             \
    if (sizeof(T) == 1) vld_b_l_xx_m(Vd, s, t); \
    if (sizeof(T) == 2) vld_h_l_xx_m(Vd, s, t); \
    if (sizeof(T) == 4) vld_w_l_xx_m(Vd, s, t); \
  }

#define vst_l_xx_m(T, Vd, s, t)                 \
  {                                             \
    if (sizeof(T) == 1) vst_b_l_xx_m(Vd, s, t); \
    if (sizeof(T) == 2) vst_h_l_xx_m(Vd, s, t); \
    if (sizeof(T) == 4) vst_w_l_xx_m(Vd, s, t); \
  }

#define vld_s_xx(T, Vd, s, t)                 \
  {                                           \
    if (sizeof(T) == 1) vld_b_s_xx(Vd, s, t); \
    if (sizeof(T) == 2) vld_h_s_xx(Vd, s, t); \
    if (sizeof(T) == 4) vld_w_s_xx(Vd, s, t); \
  }

#define vst_s_xx(T, Vd, s, t)                 \
  {                                           \
    if (sizeof(T) == 1) vst_b_s_xx(Vd, s, t); \
    if (sizeof(T) == 2) vst_h_s_xx(Vd, s, t); \
    if (sizeof(T) == 4) vst_w_s_xx(Vd, s, t); \
  }

#define vld_s_xx_m(T, Vd, s, t)                 \
  {                                             \
    if (sizeof(T) == 1) vld_b_s_xx_m(Vd, s, t); \
    if (sizeof(T) == 2) vld_h_s_xx_m(Vd, s, t); \
    if (sizeof(T) == 4) vld_w_s_xx_m(Vd, s, t); \
  }

#define vst_s_xx_m(T, Vd, s, t)                 \
  {                                             \
    if (sizeof(T) == 1) vst_b_s_xx_m(Vd, s, t); \
    if (sizeof(T) == 2) vst_h_s_xx_m(Vd, s, t); \
    if (sizeof(T) == 4) vst_w_s_xx_m(Vd, s, t); \
  }

#define vdup_x(T, Vd, s)                 \
  {                                      \
    if (sizeof(T) == 1) vdup_b_x(Vd, s); \
    if (sizeof(T) == 2) vdup_h_x(Vd, s); \
    if (sizeof(T) == 4) vdup_w_x(Vd, s); \
  }

#define vdup_x_m(T, Vd, s)                 \
  {                                        \
    if (sizeof(T) == 1) vdup_b_x_m(Vd, s); \
    if (sizeof(T) == 2) vdup_h_x_m(Vd, s); \
    if (sizeof(T) == 4) vdup_w_x_m(Vd, s); \
  }

#define test_alu_b_v(op, in0, ref)                                         \
  {                                                                        \
    uint8_t dut[VLENB] __attribute__((aligned(64))) = {0xcc};              \
    vdup_b_x(v0, in0);                                                     \
    __asm__ __volatile_always__(ARGS_F_A_A(op, v2, v0));                   \
    vst_b_x(v2, dut);                                                      \
    if (ref != dut[0]) {                                                   \
      printf("**error(%d)[%s] %02x : %02x %02x\n", __LINE__, op, in0, ref, \
             dut[0]);                                                      \
      exit(-1);                                                            \
    }                                                                      \
  }

#define test_alu_h_v(op, in0, ref)                                         \
  {                                                                        \
    uint16_t dut[VLENH] __attribute__((aligned(64))) = {0xcccc};           \
    vdup_h_x(v0, in0);                                                     \
    __asm__ __volatile_always__(ARGS_F_A_A(op, v2, v0));                   \
    vst_h_x(v2, dut);                                                      \
    if (ref != dut[0]) {                                                   \
      printf("**error(%d)[%s] %04x : %04x %04x\n", __LINE__, op, in0, ref, \
             dut[0]);                                                      \
      exit(-1);                                                            \
    }                                                                      \
  }

#define test_alu_w_v(op, in0, ref)                                          \
  {                                                                         \
    uint32_t dut[VLENW] __attribute__((aligned(64))) = {0xcccccccc};        \
    vdup_w_x(v0, in0);                                                      \
    __asm__ __volatile_always__(ARGS_F_A_A(op, v2, v0));                    \
    vst_w_x(v2, dut);                                                       \
    if (ref != dut[0]) {                                                    \
      printf("**error(%d)[%s] %08x : %08x %08lx\n", __LINE__, op, in0, ref, \
             dut[0]);                                                       \
      exit(-1);                                                             \
    }                                                                       \
  }

#define test_alu_b_vv(op, in0, in1, ref)                                   \
  {                                                                        \
    uint8_t dut[VLENB] __attribute__((aligned(64))) = {0xcc};              \
    vdup_b_x(v0, in0);                                                     \
    vdup_b_x(v1, in1);                                                     \
    __asm__ __volatile_always__(ARGS_F_A_A_A(op, v2, v0, v1));             \
    vst_b_x(v2, dut);                                                      \
    if (ref != dut[0]) {                                                   \
      printf("**error(%d)[%s] %02x %02x : %02x %02x\n", __LINE__, op, in0, \
             in1, ref, dut[0]);                                            \
      exit(-1);                                                            \
    }                                                                      \
  }

#define test_alu_h_vv(op, in0, in1, ref)                                   \
  {                                                                        \
    uint16_t dut[VLENH] __attribute__((aligned(64))) = {0xcccc};           \
    vdup_h_x(v0, in0);                                                     \
    vdup_h_x(v1, in1);                                                     \
    __asm__ __volatile_always__(ARGS_F_A_A_A(op, v2, v0, v1));             \
    vst_h_x(v2, dut);                                                      \
    if (ref != dut[0]) {                                                   \
      printf("**error(%d)[%s] %04x %04x : %04x %04x\n", __LINE__, op, in0, \
             in1, ref, dut[0]);                                            \
      exit(-1);                                                            \
    }                                                                      \
  }

#define test_alu_w_vv(op, in0, in1, ref)                                  \
  {                                                                       \
    uint32_t dut[VLENW] __attribute__((aligned(64))) = {0xcccccccc};      \
    vdup_w_x(v0, in0);                                                    \
    vdup_w_x(v1, in1);                                                    \
    __asm__ __volatile_always__(ARGS_F_A_A_A(op, v2, v0, v1));            \
    vst_w_x(v2, dut);                                                     \
    if (ref != dut[0]) {                                                  \
      printf("**error(%d)[%s] %08lx %08lx : %08lx %08lx\n", __LINE__, op, \
             static_cast<uint32_t>(in0), static_cast<uint32_t>(in1),      \
             static_cast<uint32_t>(ref), dut[0]);                         \
      exit(-1);                                                           \
    }                                                                     \
  }

#define test_aluw_h_vv(op, in0, in1, ref)                                  \
  {                                                                        \
    uint16_t dut[VLENH] __attribute__((aligned(64))) = {0xcccc};           \
    vdup_b_x(v0, in0);                                                     \
    vdup_b_x(v1, in1);                                                     \
    __asm__ __volatile__(ARGS_F_A_A_A(op, v2, v0, v1));                    \
    vst_h_x(v2, dut);                                                      \
    if (ref != dut[0]) {                                                   \
      printf("**error(%d)[%s] %02x %02x : %04x %04x\n", __LINE__, op, in0, \
             in1, ref, dut[0]);                                            \
      exit(-1);                                                            \
    }                                                                      \
  }

#define test_aluw_w_vv(op, in0, in1, ref)                                   \
  {                                                                         \
    uint32_t dut[VLENW] __attribute__((aligned(64))) = {0xcccccccc};        \
    vdup_h_x(v0, in0);                                                      \
    vdup_h_x(v1, in1);                                                      \
    __asm__ __volatile__(ARGS_F_A_A_A(op, v2, v0, v1));                     \
    vst_w_x(v2, dut);                                                       \
    if (ref != dut[0]) {                                                    \
      printf("**error(%d)[%s] %04x %04x : %08x %08lx\n", __LINE__, op, in0, \
             in1, ref, dut[0]);                                             \
      exit(-1);                                                             \
    }                                                                       \
  }

#define test_alu_b_vv3(op, in0, in1, in2, ref)                             \
  {                                                                        \
    uint8_t dut[VLENW] __attribute__((aligned(64))) = {0xcc};              \
    vdup_b_x(v0, in0);                                                     \
    vdup_b_x(v1, in1);                                                     \
    vdup_b_x(v2, in2);                                                     \
    __asm__ __volatile__(ARGS_F_A_A_A(op, v2, v0, v1));                    \
    vst_b_x(v2, dut);                                                      \
    if (ref != dut[0]) {                                                   \
      printf("**error(%d)[%s] %02x %02x %02x : %02x %02x\n", __LINE__, op, \
             in0, in1, in2, ref, dut[0]);                                  \
      exit(-1);                                                            \
    }                                                                      \
  }

#define test_alu_h_vv3(op, in0, in1, in2, ref)                             \
  {                                                                        \
    uint16_t dut[VLENW] __attribute__((aligned(64))) = {0xcccc};           \
    vdup_h_x(v0, in0);                                                     \
    vdup_h_x(v1, in1);                                                     \
    vdup_h_x(v2, in2);                                                     \
    __asm__ __volatile__(ARGS_F_A_A_A(op, v2, v0, v1));                    \
    vst_h_x(v2, dut);                                                      \
    if (ref != dut[0]) {                                                   \
      printf("**error(%d)[%s] %04x %04x %04x : %04x %04x\n", __LINE__, op, \
             in0, in1, in2, ref, dut[0]);                                  \
      exit(-1);                                                            \
    }                                                                      \
  }

#define test_alu_w_vv3(op, in0, in1, in2, ref)                              \
  {                                                                         \
    uint32_t dut[VLENW] __attribute__((aligned(64))) = {0xcccccccc};        \
    vdup_w_x(v0, in0);                                                      \
    vdup_w_x(v1, in1);                                                      \
    vdup_w_x(v2, in2);                                                      \
    __asm__ __volatile__(ARGS_F_A_A_A(op, v2, v0, v1));                     \
    vst_w_x(v2, dut);                                                       \
    if (ref != dut[0]) {                                                    \
      printf("**error(%d)[%s] %08x %08x %08x : %08x %08lx\n", __LINE__, op, \
             in0, in1, in2, ref, dut[0]);                                   \
      exit(-1);                                                             \
    }                                                                       \
  }

// clang-format off
#define test_alu_b_vx(op, in0, in1, ref)                                   \
  {                                                                        \
    uint8_t dut[VLENB] __attribute__((aligned(64))) = {0xcc};              \
    vdup_b_x(v0, in0);                                                     \
    __asm__ __volatile__(ARGS_F_A_A_A(op, v2, v0, %0) : : "r"(in1));       \
    vst_b_x(v2, dut);                                                      \
    if (ref != dut[0]) {                                                   \
      printf("**error(%d)[%s] %02x %02x : %02x %02x\n", __LINE__, op, in0, \
             in1, ref, dut[0]);                                            \
      exit(-1);                                                            \
    }                                                                      \
  }

#define test_alu_h_vx(op, in0, in1, ref)                                   \
  {                                                                        \
    uint16_t dut[VLENH] __attribute__((aligned(64))) = {0xcccc};           \
    vdup_h_x(v0, in0);                                                     \
    __asm__ __volatile__(ARGS_F_A_A_A(op, v2, v0, %0) : : "r"(in1));       \
    vst_h_x(v2, dut);                                                      \
    if (ref != dut[0]) {                                                   \
      printf("**error(%d)[%s] %04x %04x : %04x %04x\n", __LINE__, op, in0, \
             in1, ref, dut[0]);                                            \
      exit(-1);                                                            \
    }                                                                      \
  }

#define test_alu_w_vx(op, in0, in1, ref)                                     \
  {                                                                          \
    uint32_t dut[VLENW] __attribute__((aligned(64))) = {0xcccccccc};         \
    vdup_w_x(v0, in0);                                                       \
    __asm__ __volatile_always__(ARGS_F_A_A_A(op, v2, v0, %0) : : "r"(in1));  \
    vst_w_x(v2, dut);                                                        \
    if (ref != dut[0]) {                                                     \
      printf("**error(%d)[%s] %08x %08x : %08x %08lx\n", __LINE__, op, in0,  \
             in1, ref, dut[0]);                                              \
      exit(-1);                                                              \
    }                                                                        \
  }
// clang-format on

#endif  // TESTS_KELVIN_ISA_KELVIN_TEST_H_
