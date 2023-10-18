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

#include "tests/cv/diff.h"

#include <cstdint>
#include <cstdio>

#include "crt/kelvin.h"
#include "tests/cv/test_helper.h"

void diff_h_test() {
  constexpr int kNumCol = 640;
  uint16_t input0_row[kNumCol] __attribute__((aligned(64)));
  uint16_t input1_row[kNumCol] __attribute__((aligned(64)));
  uint16_t output_row[kNumCol] __attribute__((aligned(64)));
  krand(kNumCol, input0_row);
  krand(kNumCol, input1_row);

  kelvin::cv::diff(kNumCol, input0_row, input1_row, output_row);

  for (int i = 0; i < kNumCol; ++i) {
    const uint16_t ref_value = input0_row[i] - input1_row[i];
    if (ref_value != output_row[i]) {
      printf("**error::diff_h_test[%d] %x %x\n", i, ref_value, output_row[i]);
      exit(1);
    }
  }
}

void diff_hp_test() {
  constexpr int kNumCol = 640;
  uint16_t input0_row[kNumCol] __attribute__((aligned(64)));
  uint16_t input1_row[kNumCol] __attribute__((aligned(64)));
  uint16_t output_row[kNumCol] __attribute__((aligned(64)));
  krand(kNumCol, input0_row);
  krand(kNumCol, input1_row);

  kelvin::cv::diffp(kNumCol, input0_row, input1_row, output_row);

  for (int i = 0; i < kNumCol; ++i) {
    const uint16_t ref_value = input0_row[i] - input1_row[i];
    if (ref_value != output_row[i]) {
      printf("**error::diff_hp_test[%d,%d] %x %x\n", i / kNumCol, i % kNumCol,
             ref_value, output_row[i]);
      exit(1);
    }
  }
}

void diff_v_test() {
  constexpr int kNumCol = 640;
  uint16_t input0_row[kNumCol * 4] __attribute__((aligned(64)));
  uint16_t input1_row[kNumCol * 4] __attribute__((aligned(64)));
  uint16_t output_row[kNumCol * 4] __attribute__((aligned(64)));
  krand(kNumCol * 4, input0_row);
  krand(kNumCol * 4, input1_row);

  kelvin::cv::diff4(kNumCol, kNumCol, input0_row, input1_row, output_row);

  for (int i = 0; i < kNumCol * 4; ++i) {
    const uint16_t ref_value = input0_row[i] - input1_row[i];
    if (ref_value != output_row[i]) {
      printf("**error::diff_v_test[%d,%d] %x %x\n", i / kNumCol, i % kNumCol,
             ref_value, output_row[i]);
      exit(1);
    }
  }
}

void diff_vp_test() {
  constexpr int kNumCol = 640;
  uint16_t input0_row[kNumCol * 4] __attribute__((aligned(64)));
  uint16_t input1_row[kNumCol * 4] __attribute__((aligned(64)));
  uint16_t output_row[kNumCol * 4] __attribute__((aligned(64)));
  krand(kNumCol * 4, input0_row);
  krand(kNumCol * 4, input1_row);

  kelvin::cv::diff4p(kNumCol, kNumCol, input0_row, input1_row, output_row);

  for (int i = 0; i < kNumCol * 4; ++i) {
    const uint16_t ref_value = input0_row[i] - input1_row[i];
    if (ref_value != output_row[i]) {
      printf("**error::diff_vp_test[%d,%d] %x %x\n", i / kNumCol, i % kNumCol,
             ref_value, output_row[i]);
      exit(1);
    }
  }
}

int main() {
  diff_h_test();
  diff_hp_test();
  diff_v_test();
  diff_vp_test();
  return 0;
}
