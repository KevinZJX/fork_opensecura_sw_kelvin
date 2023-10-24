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

#include "crt/kelvin.h"

// clang-format off
#define test_alu_x(op, in0, ref)                                            \
  {                                                                         \
    uint32_t dut;                                                           \
    __asm__ __volatile_always__(ARGS_F_A_A(op, %0, %1)                      \
                                : "=r"(dut)                                 \
                                : "r"(in0));                                \
    if (ref != dut) {                                                       \
      printf("**error(%d)[%s] %08x : %08x %08lx\n", __LINE__, op, in0, ref, \
             dut);                                                          \
      exit(-1);                                                             \
    }                                                                       \
  }

#define test_alu_xx(op, in0, in1, ref)                                      \
  {                                                                         \
    uint32_t dut;                                                           \
    __asm__ __volatile_always__(ARGS_F_A_A_A(op, %0, %1, %2)                \
                                : "=r"(dut)                                 \
                                : "r"(in0), "r"(in1));                      \
    if (ref != dut) {                                                       \
      printf("**error(%d)[%s] %08x %08x : %08x %08lx\n", __LINE__, op, in0, \
             in1, ref, dut);                                                \
      exit(-1);                                                             \
    }                                                                       \
  }

static void div_under_mul() {
  // Check write port collision of divide and multiply units.
  constexpr int ref = 14;
  int dut, n = 100, d = 7;
  __asm__ volatile(ARGS_F_A_A_A("div", %0, %1, %2)
                   : "=r"(dut)
                   : "r"(n), "r"(d));
  // clang-format on
  __asm__ volatile("mul s0, s1, s1");
  __asm__ volatile("mul s2, s3, s3");
  __asm__ volatile("mul s4, s5, s5");
  __asm__ volatile("mul s6, s7, s7");
  __asm__ volatile("mul s8, s9, s9");
  __asm__ volatile("mul s0, s1, s1");
  __asm__ volatile("mul s2, s3, s3");
  __asm__ volatile("mul s4, s5, s5");
  __asm__ volatile("mul s6, s7, s7");
  __asm__ volatile("mul s8, s9, s9");
  __asm__ volatile("mul s0, s1, s1");
  __asm__ volatile("mul s2, s3, s3");
  __asm__ volatile("mul s4, s5, s5");
  __asm__ volatile("mul s6, s7, s7");
  __asm__ volatile("mul s8, s9, s9");
  __asm__ volatile("mul s0, s1, s1");
  __asm__ volatile("mul s2, s3, s3");
  __asm__ volatile("mul s4, s5, s5");
  __asm__ volatile("mul s6, s7, s7");
  __asm__ volatile("mul s8, s9, s9");

  if (ref != dut) {
    printf("**error alu::div_under_mul %d %d\n", ref, dut);
    exit(-1);
  }
}

int main() {
  test_alu_xx("and", 0x55ffcc5a, 0xaa015566, 0x00014442);

  test_alu_x("not", 0x55ffcc5a, 0xaa0033a5);
  test_alu_x("not", 0xa05f1248, 0x5fa0edb7);

  test_alu_xx("or", 0x55555555, 0xaaaaaaaa, 0xffffffff);
  test_alu_xx("or", 0xa05f1248, 0x5fa02481, 0xffff36c9);
  test_alu_xx("or", 0x12345678, 0x87654321, 0x97755779);

  test_alu_xx("xor", 0x55555555, 0xaaaaaaaa, 0xffffffff);
  test_alu_xx("xor", 0x55555555, 0x5555ffff, 0x0000aaaa);
  test_alu_xx("xor", 0xa05f1248, 0x5fa02481, 0xffff36c9);
  test_alu_xx("xor", 0x12345678, 0x87654321, 0x95511559);

  test_alu_xx("slt", 0x00000000, 0x00000000, 0);
  test_alu_xx("slt", 0x00000000, 0x55555555, 1);
  test_alu_xx("slt", 0x00000000, 0x7fffffff, 1);
  test_alu_xx("slt", 0x00000000, 0x80000000, 0);
  test_alu_xx("slt", 0x00000000, 0xcccccccc, 0);
  test_alu_xx("slt", 0x00000000, 0xffffffff, 0);
  test_alu_xx("slt", 0x55555555, 0x00000000, 0);
  test_alu_xx("slt", 0x55555555, 0x55555555, 0);
  test_alu_xx("slt", 0x55555555, 0x7fffffff, 1);
  test_alu_xx("slt", 0x55555555, 0x80000000, 0);
  test_alu_xx("slt", 0x55555555, 0xcccccccc, 0);
  test_alu_xx("slt", 0x55555555, 0xffffffff, 0);
  test_alu_xx("slt", 0x7fffffff, 0x00000000, 0);
  test_alu_xx("slt", 0x7fffffff, 0x55555555, 0);
  test_alu_xx("slt", 0x7fffffff, 0x7fffffff, 0);
  test_alu_xx("slt", 0x7fffffff, 0x80000000, 0);
  test_alu_xx("slt", 0x7fffffff, 0xcccccccc, 0);
  test_alu_xx("slt", 0x7fffffff, 0xffffffff, 0);
  test_alu_xx("slt", 0x80000000, 0x00000000, 1);
  test_alu_xx("slt", 0x80000000, 0x55555555, 1);
  test_alu_xx("slt", 0x80000000, 0x7fffffff, 1);
  test_alu_xx("slt", 0x80000000, 0x80000000, 0);
  test_alu_xx("slt", 0x80000000, 0xcccccccc, 1);
  test_alu_xx("slt", 0x80000000, 0xffffffff, 1);
  test_alu_xx("slt", 0xcccccccc, 0x00000000, 1);
  test_alu_xx("slt", 0xcccccccc, 0x55555555, 1);
  test_alu_xx("slt", 0xcccccccc, 0x7fffffff, 1);
  test_alu_xx("slt", 0xcccccccc, 0x80000000, 0);
  test_alu_xx("slt", 0xcccccccc, 0xcccccccc, 0);
  test_alu_xx("slt", 0xcccccccc, 0xffffffff, 1);
  test_alu_xx("slt", 0xffffffff, 0x00000000, 1);
  test_alu_xx("slt", 0xffffffff, 0x55555555, 1);
  test_alu_xx("slt", 0xffffffff, 0x7fffffff, 1);
  test_alu_xx("slt", 0xffffffff, 0x80000000, 0);
  test_alu_xx("slt", 0xffffffff, 0xcccccccc, 0);
  test_alu_xx("slt", 0xffffffff, 0xffffffff, 0);

  test_alu_xx("sltu", 0x00000000, 0x00000000, 0);
  test_alu_xx("sltu", 0x00000000, 0x55555555, 1);
  test_alu_xx("sltu", 0x00000000, 0x7fffffff, 1);
  test_alu_xx("sltu", 0x00000000, 0x80000000, 1);
  test_alu_xx("sltu", 0x00000000, 0xcccccccc, 1);
  test_alu_xx("sltu", 0x00000000, 0xffffffff, 1);
  test_alu_xx("sltu", 0x55555555, 0x00000000, 0);
  test_alu_xx("sltu", 0x55555555, 0x55555555, 0);
  test_alu_xx("sltu", 0x55555555, 0x7fffffff, 1);
  test_alu_xx("sltu", 0x55555555, 0x80000000, 1);
  test_alu_xx("sltu", 0x55555555, 0xcccccccc, 1);
  test_alu_xx("sltu", 0x55555555, 0xffffffff, 1);
  test_alu_xx("sltu", 0x7fffffff, 0x00000000, 0);
  test_alu_xx("sltu", 0x7fffffff, 0x55555555, 0);
  test_alu_xx("sltu", 0x7fffffff, 0x7fffffff, 0);
  test_alu_xx("sltu", 0x7fffffff, 0x80000000, 1);
  test_alu_xx("sltu", 0x7fffffff, 0xcccccccc, 1);
  test_alu_xx("sltu", 0x7fffffff, 0xffffffff, 1);
  test_alu_xx("sltu", 0x80000000, 0x00000000, 0);
  test_alu_xx("sltu", 0x80000000, 0x55555555, 0);
  test_alu_xx("sltu", 0x80000000, 0x7fffffff, 0);
  test_alu_xx("sltu", 0x80000000, 0x80000000, 0);
  test_alu_xx("sltu", 0x80000000, 0xcccccccc, 1);
  test_alu_xx("sltu", 0x80000000, 0xffffffff, 1);
  test_alu_xx("sltu", 0xcccccccc, 0x00000000, 0);
  test_alu_xx("sltu", 0xcccccccc, 0x55555555, 0);
  test_alu_xx("sltu", 0xcccccccc, 0x7fffffff, 0);
  test_alu_xx("sltu", 0xcccccccc, 0x80000000, 0);
  test_alu_xx("sltu", 0xcccccccc, 0xcccccccc, 0);
  test_alu_xx("sltu", 0xcccccccc, 0xffffffff, 1);
  test_alu_xx("sltu", 0xffffffff, 0x00000000, 0);
  test_alu_xx("sltu", 0xffffffff, 0x55555555, 0);
  test_alu_xx("sltu", 0xffffffff, 0x7fffffff, 0);
  test_alu_xx("sltu", 0xffffffff, 0x80000000, 0);
  test_alu_xx("sltu", 0xffffffff, 0xcccccccc, 0);
  test_alu_xx("sltu", 0xffffffff, 0xffffffff, 0);

  test_alu_xx("sgt", 0x00000000, 0x00000000, 0);
  test_alu_xx("sgt", 0x00000000, 0x55555555, 0);
  test_alu_xx("sgt", 0x00000000, 0x7fffffff, 0);
  test_alu_xx("sgt", 0x00000000, 0x80000000, 1);
  test_alu_xx("sgt", 0x00000000, 0xcccccccc, 1);
  test_alu_xx("sgt", 0x00000000, 0xffffffff, 1);
  test_alu_xx("sgt", 0x55555555, 0x00000000, 1);
  test_alu_xx("sgt", 0x55555555, 0x55555555, 0);
  test_alu_xx("sgt", 0x55555555, 0x7fffffff, 0);
  test_alu_xx("sgt", 0x55555555, 0x80000000, 1);
  test_alu_xx("sgt", 0x55555555, 0xcccccccc, 1);
  test_alu_xx("sgt", 0x55555555, 0xffffffff, 1);
  test_alu_xx("sgt", 0x7fffffff, 0x00000000, 1);
  test_alu_xx("sgt", 0x7fffffff, 0x55555555, 1);
  test_alu_xx("sgt", 0x7fffffff, 0x7fffffff, 0);
  test_alu_xx("sgt", 0x7fffffff, 0x80000000, 1);
  test_alu_xx("sgt", 0x7fffffff, 0xcccccccc, 1);
  test_alu_xx("sgt", 0x7fffffff, 0xffffffff, 1);
  test_alu_xx("sgt", 0x80000000, 0x00000000, 0);
  test_alu_xx("sgt", 0x80000000, 0x55555555, 0);
  test_alu_xx("sgt", 0x80000000, 0x7fffffff, 0);
  test_alu_xx("sgt", 0x80000000, 0x80000000, 0);
  test_alu_xx("sgt", 0x80000000, 0xcccccccc, 0);
  test_alu_xx("sgt", 0x80000000, 0xffffffff, 0);
  test_alu_xx("sgt", 0xcccccccc, 0x00000000, 0);
  test_alu_xx("sgt", 0xcccccccc, 0x55555555, 0);
  test_alu_xx("sgt", 0xcccccccc, 0x7fffffff, 0);
  test_alu_xx("sgt", 0xcccccccc, 0x80000000, 1);
  test_alu_xx("sgt", 0xcccccccc, 0xcccccccc, 0);
  test_alu_xx("sgt", 0xcccccccc, 0xffffffff, 0);
  test_alu_xx("sgt", 0xffffffff, 0x00000000, 0);
  test_alu_xx("sgt", 0xffffffff, 0x55555555, 0);
  test_alu_xx("sgt", 0xffffffff, 0x7fffffff, 0);
  test_alu_xx("sgt", 0xffffffff, 0x80000000, 1);
  test_alu_xx("sgt", 0xffffffff, 0xcccccccc, 1);
  test_alu_xx("sgt", 0xffffffff, 0xffffffff, 0);

  test_alu_xx("sgtu", 0x00000000, 0x00000000, 0);
  test_alu_xx("sgtu", 0x00000000, 0x55555555, 0);
  test_alu_xx("sgtu", 0x00000000, 0x7fffffff, 0);
  test_alu_xx("sgtu", 0x00000000, 0x80000000, 0);
  test_alu_xx("sgtu", 0x00000000, 0xcccccccc, 0);
  test_alu_xx("sgtu", 0x00000000, 0xffffffff, 0);
  test_alu_xx("sgtu", 0x55555555, 0x00000000, 1);
  test_alu_xx("sgtu", 0x55555555, 0x55555555, 0);
  test_alu_xx("sgtu", 0x55555555, 0x7fffffff, 0);
  test_alu_xx("sgtu", 0x55555555, 0x80000000, 0);
  test_alu_xx("sgtu", 0x55555555, 0xcccccccc, 0);
  test_alu_xx("sgtu", 0x55555555, 0xffffffff, 0);
  test_alu_xx("sgtu", 0x7fffffff, 0x00000000, 1);
  test_alu_xx("sgtu", 0x7fffffff, 0x55555555, 1);
  test_alu_xx("sgtu", 0x7fffffff, 0x7fffffff, 0);
  test_alu_xx("sgtu", 0x7fffffff, 0x80000000, 0);
  test_alu_xx("sgtu", 0x7fffffff, 0xcccccccc, 0);
  test_alu_xx("sgtu", 0x7fffffff, 0xffffffff, 0);
  test_alu_xx("sgtu", 0x80000000, 0x00000000, 1);
  test_alu_xx("sgtu", 0x80000000, 0x55555555, 1);
  test_alu_xx("sgtu", 0x80000000, 0x7fffffff, 1);
  test_alu_xx("sgtu", 0x80000000, 0x80000000, 0);
  test_alu_xx("sgtu", 0x80000000, 0xcccccccc, 0);
  test_alu_xx("sgtu", 0x80000000, 0xffffffff, 0);
  test_alu_xx("sgtu", 0xcccccccc, 0x00000000, 1);
  test_alu_xx("sgtu", 0xcccccccc, 0x55555555, 1);
  test_alu_xx("sgtu", 0xcccccccc, 0x7fffffff, 1);
  test_alu_xx("sgtu", 0xcccccccc, 0x80000000, 1);
  test_alu_xx("sgtu", 0xcccccccc, 0xcccccccc, 0);
  test_alu_xx("sgtu", 0xcccccccc, 0xffffffff, 0);
  test_alu_xx("sgtu", 0xffffffff, 0x00000000, 1);
  test_alu_xx("sgtu", 0xffffffff, 0x55555555, 1);
  test_alu_xx("sgtu", 0xffffffff, 0x7fffffff, 1);
  test_alu_xx("sgtu", 0xffffffff, 0x80000000, 1);
  test_alu_xx("sgtu", 0xffffffff, 0xcccccccc, 1);
  test_alu_xx("sgtu", 0xffffffff, 0xffffffff, 0);

  test_alu_xx("sll", 0x23456789, 0x00, 0x23456789);
  test_alu_xx("sll", 0x23456789, 0x01, 0x468acf12);
  test_alu_xx("sll", 0x23456789, 0x02, 0x8d159e24);
  test_alu_xx("sll", 0x23456789, 0x04, 0x34567890);
  test_alu_xx("sll", 0x23456789, 0x08, 0x45678900);
  test_alu_xx("sll", 0x23456789, 0x10, 0x67890000);
  test_alu_xx("sll", 0x23456789, 0x1f, 0x80000000);
  test_alu_xx("sll", 0x23456789, 0xe0, 0x23456789);

  test_alu_xx("sra", 0x23456789, 0x00, 0x23456789);
  test_alu_xx("sra", 0x23456789, 0x01, 0x11a2b3c4);
  test_alu_xx("sra", 0x23456789, 0x02, 0x08d159e2);
  test_alu_xx("sra", 0x23456789, 0x04, 0x02345678);
  test_alu_xx("sra", 0x23456789, 0x08, 0x00234567);
  test_alu_xx("sra", 0x23456789, 0x10, 0x00002345);
  test_alu_xx("sra", 0x23456789, 0x1f, 0x00000000);
  test_alu_xx("sra", 0x23456789, 0xe0, 0x23456789);
  test_alu_xx("sra", 0x98765432, 0x00, 0x98765432);
  test_alu_xx("sra", 0x98765432, 0x01, 0xcc3b2a19);
  test_alu_xx("sra", 0x98765432, 0x02, 0xe61d950c);
  test_alu_xx("sra", 0x98765432, 0x04, 0xf9876543);
  test_alu_xx("sra", 0x98765432, 0x08, 0xff987654);
  test_alu_xx("sra", 0x98765432, 0x10, 0xffff9876);
  test_alu_xx("sra", 0x98765432, 0x1f, 0xffffffff);
  test_alu_xx("sra", 0x98765432, 0xe0, 0x98765432);

  test_alu_xx("srl", 0x23456789, 0x00, 0x23456789);
  test_alu_xx("srl", 0x23456789, 0x01, 0x11a2b3c4);
  test_alu_xx("srl", 0x23456789, 0x02, 0x08d159e2);
  test_alu_xx("srl", 0x23456789, 0x04, 0x02345678);
  test_alu_xx("srl", 0x23456789, 0x08, 0x00234567);
  test_alu_xx("srl", 0x23456789, 0x10, 0x00002345);
  test_alu_xx("srl", 0x23456789, 0x1f, 0x00000000);
  test_alu_xx("srl", 0x23456789, 0xe0, 0x23456789);
  test_alu_xx("srl", 0x98765432, 0x00, 0x98765432);
  test_alu_xx("srl", 0x98765432, 0x01, 0x4c3b2a19);
  test_alu_xx("srl", 0x98765432, 0x02, 0x261d950c);
  test_alu_xx("srl", 0x98765432, 0x04, 0x09876543);
  test_alu_xx("srl", 0x98765432, 0x08, 0x00987654);
  test_alu_xx("srl", 0x98765432, 0x10, 0x00009876);
  test_alu_xx("srl", 0x98765432, 0x1f, 0x00000001);
  test_alu_xx("srl", 0x98765432, 0xe0, 0x98765432);

  test_alu_xx("mul", 0x00000000, 0x00000000, 0x00000000);
  test_alu_xx("mul", 0x55555555, 0x00000000, 0x00000000);
  test_alu_xx("mul", 0x00000002, 0x00000003, 0x00000006);
  test_alu_xx("mul", 0x12345678, 0x00000006, 0x6d3a06d0);
  test_alu_xx("mul", 0x12345678, 0x00000009, 0xa3d70a38);
  test_alu_xx("mul", 0x55555555, 0x00000003, 0xffffffff);
  test_alu_xx("mul", 0x55555555, 0x00000005, 0xaaaaaaa9);
  test_alu_xx("mul", 0x23456789, 0x02305670, 0x3ad551f0);
  test_alu_xx("mul", 0x543210fe, 0x12345678, 0x98c54b10);
  test_alu_xx("mul", 0x543210fe, 0x89abcdef, 0xfa034322);
  test_alu_xx("mul", 0xedcba987, 0x12345678, 0xcfd6d148);
  test_alu_xx("mul", 0xedcba987, 0x89abcdef, 0x94116009);
  test_alu_xx("mul", 0xcba98765, 0x00000007, 0x91a2b3c3);
  test_alu_xx("mul", 0xcba98765, 0x98765432, 0xc81795ba);
  test_alu_xx("mul", 0xcba98765, 0xdef34567, 0xbd92b2a3);

  test_alu_xx("mulh", 0x00000000, 0x00000000, 0x00000000);
  test_alu_xx("mulh", 0x55555555, 0x00000000, 0x00000000);
  test_alu_xx("mulh", 0x00000002, 0x00000003, 0x00000000);
  test_alu_xx("mulh", 0x12345678, 0x00000006, 0x00000000);
  test_alu_xx("mulh", 0x12345678, 0x00000009, 0x00000000);
  test_alu_xx("mulh", 0x55555555, 0x00000003, 0x00000000);
  test_alu_xx("mulh", 0x55555555, 0x00000005, 0x00000001);
  test_alu_xx("mulh", 0x23456789, 0x02305670, 0x004d33bb);
  test_alu_xx("mulh", 0x543210fe, 0x12345678, 0x05fcbbcd);
  test_alu_xx("mulh", 0x543210fe, 0x89abcdef, 0xd9153b45);
  test_alu_xx("mulh", 0xedcba987, 0x12345678, 0xfeb49923);
  test_alu_xx("mulh", 0xedcba987, 0x89abcdef, 0x086a1c97);
  test_alu_xx("mulh", 0xcba98765, 0x00000007, 0xfffffffe);
  test_alu_xx("mulh", 0xcba98765, 0x98765432, 0x152aefec);
  test_alu_xx("mulh", 0xcba98765, 0xdef34567, 0x06c1bfbf);

  test_alu_xx("mulhu", 0x00000000, 0x00000000, 0x00000000);
  test_alu_xx("mulhu", 0x55555555, 0x00000000, 0x00000000);
  test_alu_xx("mulhu", 0x00000002, 0x00000003, 0x00000000);
  test_alu_xx("mulhu", 0x12345678, 0x00000006, 0x00000000);
  test_alu_xx("mulhu", 0x12345678, 0x00000009, 0x00000000);
  test_alu_xx("mulhu", 0x55555555, 0x00000003, 0x00000000);
  test_alu_xx("mulhu", 0x55555555, 0x00000005, 0x00000001);
  test_alu_xx("mulhu", 0x23456789, 0x02305670, 0x004d33bb);
  test_alu_xx("mulhu", 0x543210fe, 0x12345678, 0x05fcbbcd);
  test_alu_xx("mulhu", 0x543210fe, 0x89abcdef, 0x2d474c43);
  test_alu_xx("mulhu", 0xedcba987, 0x12345678, 0x10e8ef9b);
  test_alu_xx("mulhu", 0xedcba987, 0x89abcdef, 0x7fe1940d);
  test_alu_xx("mulhu", 0xcba98765, 0x00000007, 0x00000005);
  test_alu_xx("mulhu", 0xcba98765, 0x98765432, 0x794acb83);
  test_alu_xx("mulhu", 0xcba98765, 0xdef34567, 0xb15e8c8b);

  test_alu_xx("div", 0x00000000, 0x00000000, 0xffffffff);
  test_alu_xx("div", 0x00000000, 0x00000001, 0x00000000);
  test_alu_xx("div", 0x00000001, 0x00000000, 0xffffffff);
  test_alu_xx("div", 0x00000001, 0x00000001, 0x00000001);
  test_alu_xx("div", 0x00001234, 0x00000000, 0xffffffff);
  test_alu_xx("div", 0xcdba9876, 0x00000000, 0xffffffff);
  test_alu_xx("div", 0x00000064, 0x0000000a, 0x0000000a);
  test_alu_xx("div", 0x00000067, 0x0000000a, 0x0000000a);
  test_alu_xx("div", 0x7fffffff, 0xffffffff, 0x80000001);
  test_alu_xx("div", 0xffffffff, 0x7fffffff, 0x00000000);
  test_alu_xx("div", 0x80000000, 0xffffffff, 0x80000000);
  test_alu_xx("div", 0xffffffff, 0x80000000, 0x00000000);
  test_alu_xx("div", 0x12345678, 0x00004567, 0x00004326);
  test_alu_xx("div", 0xcdba9876, 0x00004567, 0xffff4692);
  test_alu_xx("div", 0x12345678, 0xffffba99, 0xffffbcda);
  test_alu_xx("div", 0xcdba9876, 0xffffba99, 0x0000b96e);

  test_alu_xx("divu", 0x00000000, 0x00000000, 0xffffffff);
  test_alu_xx("divu", 0x00000000, 0x00000001, 0x00000000);
  test_alu_xx("divu", 0x00000001, 0x00000000, 0xffffffff);
  test_alu_xx("divu", 0x00000001, 0x00000001, 0x00000001);
  test_alu_xx("divu", 0x00001234, 0x00000000, 0xffffffff);
  test_alu_xx("divu", 0xcdba9876, 0x00000000, 0xffffffff);
  test_alu_xx("divu", 0x00000064, 0x0000000a, 0x0000000a);
  test_alu_xx("divu", 0x00000067, 0x0000000a, 0x0000000a);
  test_alu_xx("divu", 0x7fffffff, 0xffffffff, 0x00000000);
  test_alu_xx("divu", 0xffffffff, 0x7fffffff, 0x00000002);
  test_alu_xx("divu", 0x80000000, 0xffffffff, 0x00000000);
  test_alu_xx("divu", 0xffffffff, 0x80000000, 0x00000001);
  test_alu_xx("divu", 0x12345678, 0x00004567, 0x00004326);
  test_alu_xx("divu", 0xcdba9876, 0x00004567, 0x0002f6db);
  test_alu_xx("divu", 0x12345678, 0xffffba99, 0x00000000);
  test_alu_xx("divu", 0xcdba9876, 0xffffba99, 0x00000000);

  test_alu_xx("rem", 0x00000000, 0x00000000, 0x00000000);
  test_alu_xx("rem", 0x00000000, 0x00000001, 0x00000000);
  test_alu_xx("rem", 0x00000001, 0x00000000, 0x00000001);
  test_alu_xx("rem", 0x00000001, 0x00000001, 0x00000000);
  test_alu_xx("rem", 0x00001234, 0x00000000, 0x00001234);
  test_alu_xx("rem", 0xcdba9876, 0x00000000, 0xcdba9876);
  test_alu_xx("rem", 0x00000064, 0x0000000a, 0x00000000);
  test_alu_xx("rem", 0x00000067, 0x0000000a, 0x00000003);
  test_alu_xx("rem", 0x7fffffff, 0xffffffff, 0x00000000);
  test_alu_xx("rem", 0xffffffff, 0x7fffffff, 0xffffffff);
  test_alu_xx("rem", 0x80000000, 0xffffffff, 0x00000000);
  test_alu_xx("rem", 0xffffffff, 0x80000000, 0xffffffff);
  test_alu_xx("rem", 0x12345678, 0x00004567, 0x0000142e);
  test_alu_xx("rem", 0xcdba9876, 0x00004567, 0xffffd9b8);
  test_alu_xx("rem", 0x12345678, 0xffffba99, 0x0000142e);
  test_alu_xx("rem", 0xcdba9876, 0xffffba99, 0xffffd9b8);

  test_alu_xx("remu", 0x00000000, 0x00000000, 0x00000000);
  test_alu_xx("remu", 0x00000000, 0x00000001, 0x00000000);
  test_alu_xx("remu", 0x00000001, 0x00000000, 0x00000001);
  test_alu_xx("remu", 0x00000001, 0x00000001, 0x00000000);
  test_alu_xx("remu", 0x00001234, 0x00000000, 0x00001234);
  test_alu_xx("remu", 0xcdba9876, 0x00000000, 0xcdba9876);
  test_alu_xx("remu", 0x00000064, 0x0000000a, 0x00000000);
  test_alu_xx("remu", 0x00000067, 0x0000000a, 0x00000003);
  test_alu_xx("remu", 0x7fffffff, 0xffffffff, 0x7fffffff);
  test_alu_xx("remu", 0xffffffff, 0x7fffffff, 0x00000001);
  test_alu_xx("remu", 0x80000000, 0xffffffff, 0x80000000);
  test_alu_xx("remu", 0xffffffff, 0x80000000, 0x7fffffff);
  test_alu_xx("remu", 0x12345678, 0x00004567, 0x0000142e);
  test_alu_xx("remu", 0xcdba9876, 0x00004567, 0x00003f59);
  test_alu_xx("remu", 0x12345678, 0xffffba99, 0x12345678);
  test_alu_xx("remu", 0xcdba9876, 0xffffba99, 0xcdba9876);

  div_under_mul();

  return 0;
}
