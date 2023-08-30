// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef TESTS_CV_DIFF_H_
#define TESTS_CV_DIFF_H_

#include <cstdint>

namespace kelvin::cv {

// DUT: diff.cc
// REF: diff_test.cc

// Stripmine horizontally one line.
void diff(int num_cols, const uint16_t* input0_row, const uint16_t* input1_row,
          uint16_t* output_row);

// Stripmine horizontally one line with stage pipelining.
void diffp(int num_cols, const uint16_t* input0_row, const uint16_t* input1_row,
           uint16_t* output_row);

// Stripmine vertically four lines.
void diff4(int num_cols, int stride, const uint16_t* input0_row,
           const uint16_t* input1_row, uint16_t* output_row);

// Stripmine vertically four lines with stage pipelining.
void diff4p(int num_cols, int stride, const uint16_t* input0_row,
            const uint16_t* input1_row, uint16_t* output_row);

};  // namespace kelvin::cv

#endif  // TESTS_CV_DIFF_H_
