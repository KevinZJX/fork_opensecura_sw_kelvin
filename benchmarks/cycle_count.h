/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#ifndef BENCHMARKS_CYCLE_COUNT_H_
#define BENCHMARKS_CYCLE_COUNT_H_

inline uint64_t mcycle_read(void) {
  uint32_t cycle_low = 0;
  uint32_t cycle_high = 0;
  uint32_t cycle_high_2 = 0;
  asm volatile(
      "1:"
      "  csrr %0, mcycleh;"  // Read `mcycleh`.
      "  csrr %1, mcycle;"   // Read `mcycle`.
      "  csrr %2, mcycleh;"  // Read `mcycleh` again.
      "  bne  %0, %2, 1b;"
      : "=r"(cycle_high), "=r"(cycle_low), "=r"(cycle_high_2)
      :);
  return static_cast<uint64_t>(cycle_high) << 32 | cycle_low;
}

#endif // #ifndef BENCHMARKS_CYCLE_COUNT_H_