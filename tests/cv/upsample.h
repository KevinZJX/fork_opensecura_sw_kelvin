// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTS_CV_UPSAMPLE_H_
#define TESTS_CV_UPSAMPLE_H_

#include <cstdint>

namespace kelvin::cv {

// DUT: upsample.cc
// REF: upsample_test.cc

void upsample(int num_output_cols, uint16_t* input0_row, uint16_t* input1_row,
              uint16_t* output0_row, uint16_t* output1_row);

};  // namespace kelvin::cv

#endif  // TESTS_CV_UPSAMPLE_H_
