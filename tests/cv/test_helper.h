// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTS_CV_TEST_HELPER_H_
#define TESTS_CV_TEST_HELPER_H_

#include <cstdint>

static uint32_t krand(void) {
  static uint32_t x = 123456789;
  static uint32_t y = 362436069;
  static uint32_t z = 521288629;
  static uint32_t w = 88675123;
  uint32_t t = x ^ (x << 11);
  x = y;
  y = z;
  z = w;
  return w = w ^ (w >> 19) ^ (t ^ (t >> 8));
}

template <typename T>
void krand(int len, T* data) {
  for (int i = 0; i < len; ++i) {
    data[i] = krand();
  }
}

#endif  // TESTS_CV_TEST_HELPER_H_
