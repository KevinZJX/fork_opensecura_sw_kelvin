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

#include "tests/kelvin_isa/kelvin_test.h"

void cmp_b_test() {
  constexpr uint8_t test_vector[] = {0x00, 0x55, 0x7f, 0x80, 0xcc, 0xff};
  for (size_t i = 0; i < sizeof(test_vector); ++i) {
    for (size_t j = 0; j < sizeof(test_vector); ++j) {
      test_alu_b_vv("veq.b.vv", test_vector[i], test_vector[j],
                    static_cast<uint8_t>(i == j));
      test_alu_b_vv("vne.b.vv", test_vector[i], test_vector[j],
                    static_cast<uint8_t>(i != j));
      test_alu_b_vv("vlt.b.vv", test_vector[i], test_vector[j],
                    static_cast<uint8_t>(static_cast<int8_t>(test_vector[i]) <
                                         static_cast<int8_t>(test_vector[j])));
      test_alu_b_vv("vlt.b.u.vv", test_vector[i], test_vector[j],
                    static_cast<uint8_t>(test_vector[i] < test_vector[j]));
      test_alu_b_vv("vle.b.vv", test_vector[i], test_vector[j],
                    static_cast<uint8_t>(static_cast<int8_t>(test_vector[i]) <=
                                         static_cast<int8_t>(test_vector[j])));
      test_alu_b_vv("vle.b.u.vv", test_vector[i], test_vector[j],
                    static_cast<uint8_t>(test_vector[i] <= test_vector[j]));
      test_alu_b_vv("vgt.b.vv", test_vector[i], test_vector[j],
                    static_cast<uint8_t>(static_cast<int8_t>(test_vector[i]) >
                                         static_cast<int8_t>(test_vector[j])));
      test_alu_b_vv("vgt.b.u.vv", test_vector[i], test_vector[j],
                    static_cast<uint8_t>(test_vector[i] > test_vector[j]));
      test_alu_b_vv("vge.b.vv", test_vector[i], test_vector[j],
                    static_cast<uint8_t>(static_cast<int8_t>(test_vector[i]) >=
                                         static_cast<int8_t>(test_vector[j])));
      test_alu_b_vv("vge.b.u.vv", test_vector[i], test_vector[j],
                    static_cast<uint8_t>(test_vector[i] >= test_vector[j]));
    }
  }
}

void cmp_h_test() {
  constexpr uint16_t test_vector[] = {0x0000, 0x5555, 0x7fff,
                                      0x8000, 0xcccc, 0xffff};
  for (size_t i = 0; i < sizeof(test_vector) / sizeof(uint16_t); ++i) {
    for (size_t j = 0; j < sizeof(test_vector) / sizeof(uint16_t); ++j) {
      test_alu_h_vv("veq.h.vv", test_vector[i], test_vector[j],
                    static_cast<uint16_t>(i == j));
      test_alu_h_vv("vne.h.vv", test_vector[i], test_vector[j],
                    static_cast<uint16_t>(i != j));
      test_alu_h_vv(
          "vlt.h.vv", test_vector[i], test_vector[j],
          static_cast<uint16_t>(static_cast<int16_t>(test_vector[i]) <
                                static_cast<int16_t>(test_vector[j])));
      test_alu_h_vv("vlt.h.u.vv", test_vector[i], test_vector[j],
                    static_cast<uint16_t>(test_vector[i] < test_vector[j]));
      test_alu_h_vv(
          "vle.h.vv", test_vector[i], test_vector[j],
          static_cast<uint16_t>(static_cast<int16_t>(test_vector[i]) <=
                                static_cast<int16_t>(test_vector[j])));
      test_alu_h_vv("vle.h.u.vv", test_vector[i], test_vector[j],
                    static_cast<uint16_t>(test_vector[i] <= test_vector[j]));
      test_alu_h_vv(
          "vgt.h.vv", test_vector[i], test_vector[j],
          static_cast<uint16_t>(static_cast<int16_t>(test_vector[i]) >
                                static_cast<int16_t>(test_vector[j])));
      test_alu_h_vv("vgt.h.u.vv", test_vector[i], test_vector[j],
                    static_cast<uint16_t>(test_vector[i] > test_vector[j]));
      test_alu_h_vv(
          "vge.h.vv", test_vector[i], test_vector[j],
          static_cast<uint16_t>(static_cast<int16_t>(test_vector[i]) >=
                                static_cast<int16_t>(test_vector[j])));
      test_alu_h_vv("vge.h.u.vv", test_vector[i], test_vector[j],
                    static_cast<uint16_t>(test_vector[i] >= test_vector[j]));
    }
  }
}

void cmp_w_test() {
  constexpr uint32_t test_vector[] = {0x00000000, 0x55555555, 0x7fffffff,
                                      0x80000000, 0xcccccccc, 0xffffffff};
  for (size_t i = 0; i < sizeof(test_vector) / sizeof(uint32_t); ++i) {
    for (size_t j = 0; j < sizeof(test_vector) / sizeof(uint32_t); ++j) {
      test_alu_w_vv("veq.w.vv", test_vector[i], test_vector[j],
                    static_cast<uint32_t>(i == j));
      test_alu_w_vv("vne.w.vv", test_vector[i], test_vector[j],
                    static_cast<uint32_t>(i != j));
      test_alu_w_vv(
          "vlt.w.vv", test_vector[i], test_vector[j],
          static_cast<uint32_t>(static_cast<int32_t>(test_vector[i]) <
                                static_cast<int32_t>(test_vector[j])));
      test_alu_w_vv("vlt.w.u.vv", test_vector[i], test_vector[j],
                    static_cast<uint32_t>(test_vector[i] < test_vector[j]));
      test_alu_w_vv(
          "vle.w.vv", test_vector[i], test_vector[j],
          static_cast<uint32_t>(static_cast<int32_t>(test_vector[i]) <=
                                static_cast<int32_t>(test_vector[j])));
      test_alu_w_vv("vle.w.u.vv", test_vector[i], test_vector[j],
                    static_cast<uint32_t>(test_vector[i] <= test_vector[j]));
      test_alu_w_vv(
          "vgt.w.vv", test_vector[i], test_vector[j],
          static_cast<uint32_t>(static_cast<int32_t>(test_vector[i]) >
                                static_cast<int32_t>(test_vector[j])));
      test_alu_w_vv("vgt.w.u.vv", test_vector[i], test_vector[j],
                    static_cast<uint32_t>(test_vector[i] > test_vector[j]));
      test_alu_w_vv(
          "vge.w.vv", test_vector[i], test_vector[j],
          static_cast<uint32_t>(static_cast<int32_t>(test_vector[i]) >=
                                static_cast<int32_t>(test_vector[j])));
      test_alu_w_vv("vge.w.u.vv", test_vector[i], test_vector[j],
                    static_cast<uint32_t>(test_vector[i] >= test_vector[j]));
    }
  }
}

int main() {
  cmp_b_test();
  cmp_h_test();
  cmp_w_test();
  return 0;
}
