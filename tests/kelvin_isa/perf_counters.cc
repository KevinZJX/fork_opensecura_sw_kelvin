/*
 * Copyright 2024 Google LLC
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
#include <cstdint>

#include "tests/kelvin_isa/kelvin_test.h"

inline uint64_t mcycle_read(void) {
  uint32_t cycle_low = 0;
  uint32_t cycle_high = 0;
  asm volatile(
      "1:"
      "  csrr %0, mcycleh;"  // Read `mcycleh`.
      "  csrr %1, mcycle;"   // Read `mcycle`.
      : "=r"(cycle_high), "=r"(cycle_low)
      :);
  return static_cast<uint64_t>(cycle_high) << 32 | cycle_low;
}

int main(void) {
    // Set the cycle counter to 0x1ffffffff.
    asm volatile (" \
        csrwi mcycleh, 1; \
        li a0, 0xfffffffd; \
        csrrw a0, mcycle, a0;" : /* no outputs*/ : /* no inputs */ : /* clobbers */"a0");
    uint64_t cycle = mcycle_read();
    uint64_t cycle2 = mcycle_read();
    uint32_t cycle_lo, cycle_hi, cycle2_lo, cycle2_hi;
    cycle_lo = cycle & 0xFFFFFFFF;
    cycle2_lo = cycle2 & 0xFFFFFFFF;
    cycle_hi = cycle >> 32;
    cycle2_hi = cycle2 >> 32;
    if (cycle2_hi == cycle_hi) {
        printf("mcycleh did not increment\r\n");
        exit(-1);
    }
    if (cycle2_lo > cycle_lo) {
        printf("mcycle did not wrap");
        exit(-1);
    }
    return 0;
}
