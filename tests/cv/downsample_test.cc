
// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "tests/cv/downsample.h"

#include <cstdint>

#include "crt/kelvin.h"
#include "tests/cv/test_helper.h"

void downsample_test() {
  constexpr int kNumInputCols = 640;
  constexpr int kNumOutputCols = kNumInputCols / 2;
  uint16_t input0_row[kNumInputCols] __attribute__((aligned(64)));
  uint16_t input1_row[kNumInputCols] __attribute__((aligned(64)));
  uint16_t output_row[kNumOutputCols] __attribute__((aligned(64)));
  krand(kNumInputCols, input0_row);
  krand(kNumInputCols, input1_row);

  kelvin::cv::downsample(kNumOutputCols, input0_row, input1_row, output_row);

  for (int i = 0; i < kNumOutputCols; ++i) {
    const uint32_t s0 = input0_row[2 * i] + input0_row[2 * i + 1];
    const uint32_t s1 = input1_row[2 * i] + input1_row[2 * i + 1];
    const uint32_t s012 = s0 + s1 + 2;
    const uint16_t ref_value = s012 >> 2;
    if (ref_value != output_row[i]) {
      printf("**error::downsample_test[%d] %x %x\n", i, ref_value,
             output_row[i]);
      exit(-1);
    }
  }
}

int main() {
  downsample_test();
  return 0;
}
