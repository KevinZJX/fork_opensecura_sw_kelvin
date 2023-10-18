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
