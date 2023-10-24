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

#include <limits.h>

#include <cstdint>

#include "tests/kelvin_isa/kelvin_test.h"

void max_min_b_test() {
  constexpr uint8_t kTestVector[] = {0x00, 0x55, 0x7f, 0x80, 0xcc, 0xff};
  for (size_t i = 0; i < sizeof(kTestVector); ++i) {
    for (size_t j = 0; j < sizeof(kTestVector); ++j) {
      // The reference result should be unsigned type to match the dut type in
      // the test_alu_b_vv macro.
      test_alu_b_vv(
          "vmax.b.vv", kTestVector[i], kTestVector[j],
          static_cast<uint8_t>(std::max(static_cast<int8_t>(kTestVector[i]),
                                        static_cast<int8_t>(kTestVector[j])) &
                               0xff));
      test_alu_b_vv("vmax.b.u.vv", kTestVector[i], kTestVector[j],
                    (std::max(kTestVector[i], kTestVector[j]) & 0xff));
      test_alu_b_vv(
          "vmin.b.vv", kTestVector[i], kTestVector[j],
          static_cast<uint8_t>(std::min(static_cast<int8_t>(kTestVector[i]),
                                        static_cast<int8_t>(kTestVector[j])) &
                               0xff));
      test_alu_b_vv("vmin.b.u.vv", kTestVector[i], kTestVector[j],
                    (std::min(kTestVector[i], kTestVector[j]) & 0xff));
    }
  }
}

void max_min_h_test() {
  constexpr uint16_t kTestVector[] = {0x0000, 0x5555, 0x7fff,
                                      0x8000, 0xcccc, 0xffff};
  // The reference result should be unsigned type to match the dut type in
  // the test_alu_h_vv macro.
  for (size_t i = 0; i < sizeof(kTestVector) / sizeof(uint16_t); ++i) {
    for (size_t j = 0; j < sizeof(kTestVector) / sizeof(uint16_t); ++j) {
      test_alu_h_vv(
          "vmax.h.vv", kTestVector[i], kTestVector[j],
          static_cast<uint16_t>(std::max(static_cast<int16_t>(kTestVector[i]),
                                         static_cast<int16_t>(kTestVector[j])) &
                                0xffff));
      test_alu_h_vv("vmax.h.u.vv", kTestVector[i], kTestVector[j],
                    (std::max(kTestVector[i], kTestVector[j]) & 0xffff));
      test_alu_h_vv(
          "vmin.h.vv", kTestVector[i], kTestVector[j],
          static_cast<uint16_t>(std::min(static_cast<int16_t>(kTestVector[i]),
                                         static_cast<int16_t>(kTestVector[j])) &
                                0xffff));
      test_alu_h_vv("vmin.h.u.vv", kTestVector[i], kTestVector[j],
                    (std::min(kTestVector[i], kTestVector[j]) & 0xffff));
    }
  }
}

void max_min_w_test() {
  constexpr uint32_t kTestVector[] = {0x00000000, 0x55555555, 0x7fffffff,
                                      0x80000000, 0xcccccccc, 0xffffffff};
  for (size_t i = 0; i < sizeof(kTestVector) / sizeof(uint32_t); ++i) {
    for (size_t j = 0; j < sizeof(kTestVector) / sizeof(uint32_t); ++j) {
      // The reference result should be unsigned type to match the dut type in
      // the test_alu_w_vv macro.
      test_alu_w_vv("vmax.w.vv", kTestVector[i], kTestVector[j],
                    static_cast<uint32_t>(
                        std::max(static_cast<int32_t>(kTestVector[i]),
                                 static_cast<int32_t>(kTestVector[j]))));
      test_alu_w_vv("vmax.w.u.vv", kTestVector[i], kTestVector[j],
                    std::max(kTestVector[i], kTestVector[j]));
      test_alu_w_vv("vmin.w.vv", kTestVector[i], kTestVector[j],
                    static_cast<uint32_t>(
                        std::min(static_cast<int32_t>(kTestVector[i]),
                                 static_cast<int32_t>(kTestVector[j]))));
      test_alu_w_vv("vmin.w.u.vv", kTestVector[i], kTestVector[j],
                    std::min(kTestVector[i], kTestVector[j]));
    }
  }
}

int main() {
  max_min_b_test();
  max_min_h_test();
  max_min_w_test();
  return 0;
}
