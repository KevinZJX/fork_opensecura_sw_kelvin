// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

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
