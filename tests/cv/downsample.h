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

#ifndef TESTS_CV_DOWNSAMPLE_H_
#define TESTS_CV_DOWNSAMPLE_H_

#include <cstdint>

namespace kelvin::cv {

// DUT: downsample.cc
// REF: downsample_test.cc

void downsample(int num_output_cols, const uint16_t* input0_row,
                const uint16_t* input1_row, uint16_t* output_row);

};  // namespace kelvin::cv

#endif  // TESTS_CV_DOWNSAMPLE_H_
