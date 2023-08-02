// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Use greatest common denominator routine to stress test branch/modulo op
// combinations.

#include <stdint.h>
#include <stdlib.h>

#include "crt/kelvin.h"

template <typename T>
T get_gcd(const T a, const T b) {
  T mod = a % b;
  if (mod == 0) {
    return b;
  }
  return get_gcd<T>(b, mod);
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
  uint32_t uint32_a = rand_uint32_input();
  uint32_t uint32_b = rand_uint32_input();
  int32_t int32_a = rand_int32_input();
  int32_t int32_b = rand_int32_input();

  printf("unsigned numbers: a:%u, b:%u\n", uint32_a, uint32_b);
  auto uint32_gcd = (uint32_a > uint32_b)
                        ? get_gcd<uint32_t>(uint32_a, uint32_b)
                        : get_gcd<uint32_t>(uint32_b, uint32_a);
  printf("gcd: %u\n", uint32_gcd);

  if (!(uint32_a % uint32_gcd == 0 && uint32_b % uint32_gcd == 0)) {
    printf("Invalid common denominator %u for %u and %u\n", uint32_gcd,
           uint32_a, uint32_b);
    exit(-1);
  }

  printf("signed numbers: a:%d, b:%d\n", int32_a, int32_b);
  auto int32_gcd = (int32_a > int32_b) ? get_gcd<int32_t>(int32_a, int32_b)
                                       : get_gcd<int32_t>(int32_b, int32_a);
  printf("gcd: %d\n", int32_gcd);

  if (!(int32_a % int32_gcd == 0 && int32_b % int32_gcd == 0)) {
    printf("Invalid common denominator %d for %d and %d\n", int32_gcd, int32_a,
           int32_b);
    exit(-1);
  }

  return 0;
}
