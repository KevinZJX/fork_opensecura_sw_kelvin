// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Use the routine of finding the integer part of log_3 to stress test
// branch/div op combinations.

#include <stdint.h>
#include <stdlib.h>

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
  return (a > 0) ? a : 1;
}

int32_t rand_int32_input() {
  int32_t a = ((rand() & 0x7fff) << 16) |  // NOLINT(runtime/threadsafe_fn)
              (rand() & 0xffff);           // NOLINT(runtime/threadsafe_fn)
  return (a > 0) ? a : 1;
}

int main() {
  constexpr int kSeed = 1000;
  srand(kSeed);
  uint32_t uint32_input = rand_uint32_input();
  int32_t int32_input = rand_int32_input();
  auto uint32_log3 = get_int_log3<uint32_t>(uint32_input);
  printf("Unsigned input: %u, integer log3: %u\n", uint32_input, uint32_log3);
  auto test = int_pow3<uint32_t>(uint32_log3);
  // Make the check like this to prevent overflow
  if (!(test <= uint32_input && test >= uint32_input / 3)) {
    printf("Invalid log_3 %u for %u\n", uint32_log3, uint32_input);
    exit(-1);
  }
  auto int32_log3 = get_int_log3<int32_t>(int32_input);
  printf("Signed intput:%d, integer log3: %d\n", int32_input, int32_log3);
  auto test1 = int_pow3<int32_t>(int32_log3);
  if (!(test1 <= int32_input && test1 >= int32_input / 3)) {
    printf("Invalid log_3 %d for %d\n", int32_log3, int32_input);
    exit(-1);
  }
  return 0;
}
