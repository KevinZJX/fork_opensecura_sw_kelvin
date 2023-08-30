// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTS_CV_GAUSSIAN_H_
#define TESTS_CV_GAUSSIAN_H_

#include <cstdint>

namespace kelvin::cv {

// DUT: gaussian.cc
// REF: gaussian_test.cc

void gaussian(int num_output_cols, const uint16_t* input0_row,
              const uint16_t* input1_row, const uint16_t* input2_row,
              const uint16_t* input3_row, const uint16_t* input4_row,
              bool is_stripmine, uint16_t* output_row);

};  // namespace kelvin::cv

#endif  // TESTS_CV_GAUSSIAN_H_
