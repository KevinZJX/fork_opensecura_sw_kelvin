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

#ifndef TESTS_CV_EXTREMA_H_
#define TESTS_CV_EXTREMA_H_

#include <cstdint>

namespace kelvin::cv {

// DUT: extrema.cc
// REF: extrema_test.cc

void extrema(int num_cols, const int16_t *input[4][3], uint8_t *output0,
             uint8_t *output1);

};  // namespace kelvin::cv

#endif  // TESTS_CV_EXTREMA_H_
