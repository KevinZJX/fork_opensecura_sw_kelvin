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

template <int ireg>
void test_256_actr() {
  uint32_t inp[8][8] __attribute__((aligned(64)));
  uint32_t ref[8][8] __attribute__((aligned(64)));
  uint32_t dut[8][8] __attribute__((aligned(64)));

  int value = 0;
  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      inp[j][i] = (krand() << 8) | value++;
    }
  }

  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      ref[i][j] = inp[j][i];
    }
  }

  if (ireg == 0) {
    vld_w_x(v0, inp[0]);
    vld_w_x(v1, inp[1]);
    vld_w_x(v2, inp[2]);
    vld_w_x(v3, inp[3]);
    vld_w_x(v4, inp[4]);
    vld_w_x(v5, inp[5]);
    vld_w_x(v6, inp[6]);
    vld_w_x(v7, inp[7]);
  }
  if (ireg == 16) {
    vld_w_x(v16, inp[0]);
    vld_w_x(v17, inp[1]);
    vld_w_x(v18, inp[2]);
    vld_w_x(v19, inp[3]);
    vld_w_x(v20, inp[4]);
    vld_w_x(v21, inp[5]);
    vld_w_x(v22, inp[6]);
    vld_w_x(v23, inp[7]);
  }
  if (ireg == 32) {
    vld_w_x(v32, inp[0]);
    vld_w_x(v33, inp[1]);
    vld_w_x(v34, inp[2]);
    vld_w_x(v35, inp[3]);
    vld_w_x(v36, inp[4]);
    vld_w_x(v37, inp[5]);
    vld_w_x(v38, inp[6]);
    vld_w_x(v39, inp[7]);
  }
  if (ireg == 48) {
    vld_w_x(v48, inp[0]);
    vld_w_x(v49, inp[1]);
    vld_w_x(v50, inp[2]);
    vld_w_x(v51, inp[3]);
    vld_w_x(v52, inp[4]);
    vld_w_x(v53, inp[5]);
    vld_w_x(v54, inp[6]);
    vld_w_x(v55, inp[7]);
  }

  if (ireg == 0) {
    actr_v(v48, v0);
  }
  if (ireg == 16) {
    actr_v(v48, v16);
  }
  if (ireg == 32) {
    actr_v(v48, v32);
  }
  if (ireg == 48) {
    actr_v(v48, v48);
  }
  vcget(v48);

  vst_w_x_m(v48, dut[0]);
  vst_w_x_m(v52, dut[4]);

  for (int j = 0; j < 8; ++j) {
    for (int i = 0; i < 8; ++i) {
      if (ref[j][i] != dut[j][i]) {
        printf("**error actr_v[%d][%d,%d] %08lx %08lx\n", ireg, j, i, ref[j][i],
               dut[j][i]);
        exit(-1);
      }
    }
  }
}
int main() {
  test_256_actr<0>();
  test_256_actr<16>();
  test_256_actr<32>();
  test_256_actr<48>();

  return 0;
}
