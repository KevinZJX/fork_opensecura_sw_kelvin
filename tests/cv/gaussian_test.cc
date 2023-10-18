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

#include "tests/cv/gaussian.h"

#include <algorithm>
#include <cstdint>
#include <cstdio>

#include "crt/kelvin.h"
#include "tests/cv/test_helper.h"

void gaussian_test() {
  constexpr int num_cols = 640;
  uint16_t input0_row[num_cols] __attribute__((aligned(64)));
  uint16_t input1_row[num_cols] __attribute__((aligned(64)));
  uint16_t input2_row[num_cols] __attribute__((aligned(64)));
  uint16_t input3_row[num_cols] __attribute__((aligned(64)));
  uint16_t input4_row[num_cols] __attribute__((aligned(64)));
  uint16_t output_stripmined_row[num_cols] __attribute__((aligned(64))) = {0};
  uint16_t output_row[num_cols] __attribute__((aligned(64))) = {0};
  krand(num_cols, input0_row);
  krand(num_cols, input1_row);
  krand(num_cols, input2_row);
  krand(num_cols, input3_row);
  krand(num_cols, input4_row);

  kelvin::cv::gaussian(num_cols, input0_row, input1_row, input2_row, input3_row,
                       input4_row, true /*is_stripmine*/,
                       output_stripmined_row);
  kelvin::cv::gaussian(num_cols, input0_row, input1_row, input2_row, input3_row,
                       input4_row, false /*is_stripmine*/, output_row);

  for (int i = 0; i < num_cols; ++i) {
    uint32_t h[5];
    for (int j = 0; j < 5; j++) {
      int idx = std::min(num_cols - 1, std::max(0, i + j - 2));
      uint32_t v = 0;
      v += input0_row[idx];
      v += input1_row[idx] * 4;
      v += input2_row[idx] * 6;
      v += input3_row[idx] * 4;
      v += input4_row[idx];
      h[j] = v;
    }
    const uint32_t k = h[0] + h[1] * 4 + h[2] * 6 + h[3] * 4 + h[4] + 128;
    const uint16_t ref_value = k >> 8;
    if (ref_value != output_stripmined_row[i]) {
      printf("**error::stripmined gaussian[%d] %x %x\n", i, ref_value,
             output_stripmined_row[i]);
      exit(1);
    }
    if (ref_value != output_row[i]) {
      printf("**error::gaussian[%d] %x %x\n", i, ref_value, output_row[i]);
      exit(1);
    }
  }
}

int main() {
  gaussian_test();
  return 0;
}
