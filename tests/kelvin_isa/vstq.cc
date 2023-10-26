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

constexpr int kOffset = 2 * 512 / 32;

const uint32_t pattern_[512 / 32] = {
    1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16,
};

struct __attribute__((__packed__)) __attribute__((aligned(64))) {
  uint32_t force_unaligned;    // Make data not 64-byte aligned.
  uint32_t data[4 * kOffset];  // Handle quad * offset
} buffer_;

uint32_t b_buffer2_[4 * kOffset] __attribute__((aligned(64)));

template <int offset>
void check(const int i, const uint32_t ref) {
  const uint32_t dut = buffer_.data[i * offset];
  if (ref != dut) {
    printf("**error:vstq_test<%d>(%d) %ld %ld\n", offset, i, ref, dut);
    exit(-1);
  }
}

template <int base, int offset>
void check_base_shift(const int i, const uint32_t ref) {
  int vlen;
  getmaxvl_w(vlen);
  const uint32_t dut = b_buffer2_[base * vlen / 4 + i * offset * vlen / 4];
  if (ref != dut) {
    printf("**error:vstq_test_base_offset<%d, %d>(%d) %ld %ld\n", base, offset,
           i, ref, dut);
    exit(-1);
  }
}

template <int offset>
void vstq_test() {
  uint32_t* ptr = buffer_.data;
  memset(&buffer_, 0, sizeof(buffer_));
  vld_w_x(v0, pattern_);
  vstq_w_sp_xx(v0, ptr, offset);

  check<offset>(0, 1);
  check<offset>(1, 3);
  check<offset>(2, 5);
  check<offset>(3, 7);
}

template <int base, int offset>
void vstq_test_base_offset() {
  int vlen;
  getmaxvl_w(vlen);
  uint32_t* ptr = b_buffer2_ + (base * vlen / 4);
  memset(b_buffer2_, 0, sizeof(b_buffer2_));
  vld_w_x(v0, pattern_);
  vstq_w_sp_xx(v0, ptr, offset * vlen / 4);

  check_base_shift<base, offset>(0, 1);
  check_base_shift<base, offset>(1, 3);
  check_base_shift<base, offset>(2, 5);
  check_base_shift<base, offset>(3, 7);
}

int main() {
  vstq_test<kOffset + 0>();
  vstq_test<kOffset - 1>();
  vstq_test<kOffset + 1>();

  vstq_test_base_offset<0, 1>();
  vstq_test_base_offset<0, 2>();
  vstq_test_base_offset<0, 3>();
  vstq_test_base_offset<0, 4>();
  vstq_test_base_offset<0, 5>();
  vstq_test_base_offset<1, 1>();
  vstq_test_base_offset<1, 2>();
  vstq_test_base_offset<1, 3>();
  vstq_test_base_offset<1, 4>();
  vstq_test_base_offset<1, 5>();
  vstq_test_base_offset<2, 1>();
  vstq_test_base_offset<2, 2>();
  vstq_test_base_offset<2, 3>();
  vstq_test_base_offset<2, 4>();
  vstq_test_base_offset<2, 5>();
  vstq_test_base_offset<3, 1>();
  vstq_test_base_offset<3, 2>();
  vstq_test_base_offset<3, 3>();
  vstq_test_base_offset<3, 4>();
  vstq_test_base_offset<3, 5>();
  vstq_test_base_offset<4, 1>();
  vstq_test_base_offset<4, 2>();
  vstq_test_base_offset<4, 3>();
  vstq_test_base_offset<4, 4>();
  vstq_test_base_offset<4, 5>();

  return 0;
}
