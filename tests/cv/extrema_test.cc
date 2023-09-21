// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "tests/cv/extrema.h"

#include <cstdint>
#include <cstdio>

#include "crt/kelvin.h"
#include "tests/cv/test_helper.h"

enum kMode { kNone, kMinimum, kMaximum };

template <typename TComparisonOp>
bool IsPointExtrema(const int16_t *input[4][3], int layer_id, int col,
                    TComparisonOp comparison_op) {
  const int16_t center_value = input[layer_id][1][col];
  for (int layer_id_offset = -1; layer_id_offset <= 1; layer_id_offset++) {
    for (int row_id = 0; row_id < 3; row_id++) {
      for (int col_offset = -1; col_offset <= 1; col_offset++) {
        // Do not compare to input[layer_id][1][col] which is value.
        if (layer_id_offset == 0 && row_id == 1 && col_offset == 0) {
          continue;
        }
        if (!comparison_op(
                center_value,
                input[layer_id + layer_id_offset][row_id][col + col_offset])) {
          return false;
        }
      }
    }
  }
  return true;
}

void Extrema(int num_cols, const int16_t *input[4][3], uint8_t *output0,
             uint8_t *output1) {
  auto max_comp = [](int x, int y) { return x > y; };
  auto min_comp = [](int x, int y) { return x < y; };

  // Update extrema for layer 1.
  for (int col = 1; col < num_cols - 1; col++) {
    output0[col] = kMode::kNone;
    if (IsPointExtrema(input, 1, col, max_comp)) {
      output0[col] = kMaximum;
      continue;
    }
    if (IsPointExtrema(input, 1, col, min_comp)) {
      output0[col] = kMode::kMinimum;
    }
  }

  // Update extrema for layer 2.
  for (int col = 1; col < num_cols - 1; col++) {
    output1[col] = kMode::kNone;
    if (IsPointExtrema(input, 2, col, max_comp)) {
      output1[col] = kMode::kMaximum;
      continue;
    }
    if (IsPointExtrema(input, 2, col, min_comp)) {
      output1[col] = kMode::kMinimum;
    }
  }
}

void extrema_test() {
  constexpr int kNumCols = 640;

  int16_t input0_row0[kNumCols] __attribute__((aligned(64)));
  int16_t input0_row1[kNumCols] __attribute__((aligned(64)));
  int16_t input0_row2[kNumCols] __attribute__((aligned(64)));
  int16_t input1_row0[kNumCols] __attribute__((aligned(64)));
  int16_t input1_row1[kNumCols] __attribute__((aligned(64)));
  int16_t input1_row2[kNumCols] __attribute__((aligned(64)));
  int16_t input2_row0[kNumCols] __attribute__((aligned(64)));
  int16_t input2_row1[kNumCols] __attribute__((aligned(64)));
  int16_t input2_row2[kNumCols] __attribute__((aligned(64)));
  int16_t input3_row0[kNumCols] __attribute__((aligned(64)));
  int16_t input3_row1[kNumCols] __attribute__((aligned(64)));
  int16_t input3_row2[kNumCols] __attribute__((aligned(64)));

  uint8_t output0_ref[kNumCols];
  uint8_t output1_ref[kNumCols];
  uint8_t output0_dut[kNumCols];
  uint8_t output1_dut[kNumCols];

  const int16_t *input[4][3] = {{reinterpret_cast<int16_t *>(input0_row0),
                                 reinterpret_cast<int16_t *>(input0_row1),
                                 reinterpret_cast<int16_t *>(input0_row2)},
                                {reinterpret_cast<int16_t *>(input1_row0),
                                 reinterpret_cast<int16_t *>(input1_row1),
                                 reinterpret_cast<int16_t *>(input1_row2)},
                                {reinterpret_cast<int16_t *>(input2_row0),
                                 reinterpret_cast<int16_t *>(input2_row1),
                                 reinterpret_cast<int16_t *>(input2_row2)},
                                {reinterpret_cast<int16_t *>(input3_row0),
                                 reinterpret_cast<int16_t *>(input3_row1),
                                 reinterpret_cast<int16_t *>(input3_row2)}};

  krand(kNumCols, input0_row0);
  krand(kNumCols, input0_row1);
  krand(kNumCols, input0_row2);
  krand(kNumCols, input1_row0);
  krand(kNumCols, input1_row1);
  krand(kNumCols, input1_row2);
  krand(kNumCols, input2_row0);
  krand(kNumCols, input2_row1);
  krand(kNumCols, input2_row2);
  krand(kNumCols, input3_row0);
  krand(kNumCols, input3_row1);
  krand(kNumCols, input3_row2);

  Extrema(kNumCols, input, output0_ref, output1_ref);

  kelvin::cv::extrema(kNumCols, input, output0_dut, output1_dut);

  for (int i = 1; i < kNumCols - 1; ++i) {
    const uint8_t ref = output0_ref[i];
    const uint8_t dut = output0_dut[i];
    if (ref != dut) {
      printf("**error::extrema0[%d] %x %x\n", i, ref, dut);
      exit(1);
    }
  }

  for (int i = 1; i < kNumCols - 1; ++i) {
    const uint8_t ref = output1_ref[i];
    const uint8_t dut = output1_dut[i];
    if (ref != dut) {
      printf("**error::extrema1[%d] %x %x\n", i, ref, dut);
      exit(1);
    }
  }
}

int main() {
  extrema_test();
  return 0;
}
