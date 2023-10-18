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

// Use the routine of finding the integer part of log_3 to stress test
// branch/div op combinations.

#include <cstdint>
#include <cstdio>
#include <cstdlib>

#include "crt/kelvin.h"

template <typename T>
T get_int_log3(T a) {
  T int_log3 = 0;
  if (a < 3) return 0;
  // Use do/while loop to compile the code without jump
  do {
    int_log3++;
    a /= 3;
  } while (a >= 3);
  return int_log3;
}

template <typename T>
T int_pow3(T a) {
  T result = 1;
  for (T i = 0; i < a; ++i) {
    result *= 3;
  }
  return result;
}

uint32_t rand_uint32_input() {
  uint32_t a = static_cast<uint32_t>(
      ((rand() & 0xffff) << 16) |  // NOLINT(runtime/threadsafe_fn)
      (rand() & 0xffff));          // NOLINT(runtime/threadsafe_fn)
  return std::max<uint32_t>(a, 1);
}

int32_t rand_int32_input() {
  int32_t a = ((rand() & 0x7fff) << 16) |  // NOLINT(runtime/threadsafe_fn)
              (rand() & 0xffff);           // NOLINT(runtime/threadsafe_fn)
  return std::max<int32_t>(a, 1);
}

int main() {
  constexpr int kSeed = 1000;
  srand(kSeed);
  uint32_t uint32_input = rand_uint32_input();
  int32_t int32_input = rand_int32_input();
  auto uint32_log3 = get_int_log3<uint32_t>(uint32_input);
  printf("Unsigned input: %lu, integer log3: %lu\n", uint32_input, uint32_log3);
  auto test = int_pow3<uint32_t>(uint32_log3);
  // Make the check like this to prevent overflow
  if (!(test <= uint32_input && test >= uint32_input / 3)) {
    printf("Invalid log_3 %lu for %lu\n", uint32_log3, uint32_input);
    exit(-1);
  }
  auto int32_log3 = get_int_log3<int32_t>(int32_input);
  printf("Signed intput:%ld, integer log3: %ld\n", int32_input, int32_log3);
  auto test1 = int_pow3<int32_t>(int32_log3);
  if (!(test1 <= int32_input && test1 >= int32_input / 3)) {
    printf("Invalid log_3 %ld for %ld\n", int32_log3, int32_input);
    exit(-1);
  }
  return 0;
}
