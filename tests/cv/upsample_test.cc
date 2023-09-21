// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "tests/cv/upsample.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "crt/kelvin.h"
#include "tests/cv/test_helper.h"

void upsample_test() {
  constexpr int kNumOutputCols = 640;
  constexpr int kEdge = 32;            // 512 / 16
  constexpr int kPadding = kEdge * 2;  // left/right

  uint16_t input0_row[kNumOutputCols / 2 + kPadding];
  uint16_t input1_row[kNumOutputCols / 2 + kPadding];
  uint16_t output0_row[kNumOutputCols];
  uint16_t output1_row[kNumOutputCols];
  uint16_t *input0_data = input0_row + kEdge;
  uint16_t *input1_data = input1_row + kEdge;

  krand(kNumOutputCols / 2 + kPadding, input0_row);
  krand(kNumOutputCols / 2 + kPadding, input1_row);

  kelvin::cv::upsample(kNumOutputCols, input0_data, input1_data, output0_row,
                       output1_row);

  constexpr int kHalfWidth = kNumOutputCols / 2 - 1;
  for (int i = 0; i < kNumOutputCols; ++i) {
    int c1 = std::clamp(i / 2, 0, kHalfWidth);
    int c2 = std::clamp(i & 1 ? c1 + 1 : c1 - 1, 0, kHalfWidth);

    const uint32_t a = 3 * input0_data[c1] + input0_data[c2];
    const uint32_t b = 3 * input1_data[c1] + input1_data[c2];

    const uint16_t ref0_value = (a * 3 + b + 8) / 16;
    const uint16_t ref1_value = (b * 3 + a + 8) / 16;

    if (ref0_value != output0_row[i]) {
      printf("**error::upsample_test[%d,%d] %x %x\n", 0, i, ref0_value,
             output0_row[i]);
      exit(1);
    }

    if (ref1_value != output1_row[i]) {
      printf("**error::upsample_test[%d,%d] %x %x\n", 1, i, ref1_value,
             output1_row[i]);
      exit(1);
    }
  }
}

int main() {
  upsample_test();

  return 0;
}
