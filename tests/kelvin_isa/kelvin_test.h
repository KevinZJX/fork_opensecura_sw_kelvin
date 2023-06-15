// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Kelvin ISA test common header

#ifndef TESTS_KELVIN_ISA_KELVIN_TEST_H_
#define TESTS_KELVIN_ISA_KELVIN_TEST_H_

#include "crt/kelvin.h"

// Maximum storage required for parameterized machine load/store.
constexpr int VLEN = 256;  // simd register bits. Need to match the HW parameter
constexpr int VLENB = VLEN / 8;
constexpr int VLENH = VLEN / 16;
constexpr int VLENW = VLEN / 32;

#endif  // TESTS_KELVIN_ISA_KELVIN_TEST_H_
