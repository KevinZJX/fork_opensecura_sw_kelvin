// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "tests/cv/shift_gaussian.h"

#include <algorithm>
#include <cstdint>

#include "crt/kelvin.h"
#include "tests/cv/test_helper.h"

void shift_gaussian_test() {
  constexpr int kNumCols = 640;
  uint8_t input0_row[kNumCols] __attribute__((aligned(64)));
  uint8_t input1_row[kNumCols] __attribute__((aligned(64)));
  uint8_t input2_row[kNumCols] __attribute__((aligned(64)));
  uint8_t input3_row[kNumCols] __attribute__((aligned(64)));
  uint8_t input4_row[kNumCols] __attribute__((aligned(64)));
  uint16_t output_stripmined_row[kNumCols] __attribute__((aligned(64))) = {0};
  uint16_t output_row[kNumCols] __attribute__((aligned(64))) = {0};
  krand(kNumCols, input0_row);
  krand(kNumCols, input1_row);
  krand(kNumCols, input2_row);
  krand(kNumCols, input3_row);
  krand(kNumCols, input4_row);

  kelvin::cv::shift_gaussian(kNumCols, input0_row, input1_row, input2_row,
                             input3_row, input4_row, true /*is_stripmine*/,
                             output_stripmined_row);

  kelvin::cv::shift_gaussian(kNumCols, input0_row, input1_row, input2_row,
                             input3_row, input4_row, false /*is_stripmine*/,
                             output_row);

  for (int i = 0; i < kNumCols; ++i) {
    uint16_t h[5];
    for (int j = 0; j < 5; j++) {
      int idx = std::min(kNumCols - 1, std::max(0, i + j - 2));
      uint16_t v = 0;
      v += input0_row[idx];
      v += input1_row[idx] * 4;
      v += input2_row[idx] * 6;
      v += input3_row[idx] * 4;
      v += input4_row[idx];
      h[j] = v;
    }
    const uint16_t ref_value = h[0] + h[1] * 4 + h[2] * 6 + h[3] * 4 + h[4];
    if (ref_value != output_stripmined_row[i]) {
      printf("**error::stripmine shift_gaussian[%d] %x %x\n", i, ref_value,
             output_stripmined_row[i]);
      exit(1);
    }
    if (ref_value != output_row[i]) {
      printf("**error::shift_gaussian[%d] %x %x\n", i, ref_value,
             output_row[i]);
      exit(1);
    }
  }
}

int main() {
  shift_gaussian_test();
  return 0;
}
