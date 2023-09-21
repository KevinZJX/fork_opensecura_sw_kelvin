// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Kelvin helper header

#ifndef CRT_KELVIN_H_
#define CRT_KELVIN_H_

#include <math.h>
#include <stdint.h>
#include <string.h>

#include <algorithm>

#define __volatile_always__ volatile

// Helper macros for Intrinsics definitions.
#define ARGS_F_A(FN, A0) FN " " #A0 "\n"
#define ARGS_F_A_A(FN, A0, A1) FN " " #A0 ", " #A1 "\n"
#define ARGS_F_A_A_A(FN, A0, A1, A2) FN " " #A0 ", " #A1 ", " #A2 "\n"
#define ARGS_F_A_A_A_A(FN, A0, A1, A2, A3) \
  FN " " #A0 ", " #A1 ", " #A2 ", " #A3 "\n"

#include "crt/kelvin_intrinsics.h"

#define vm0 v0
#define vm1 v4
#define vm2 v8
#define vm3 v12
#define vm4 v16
#define vm5 v20
#define vm6 v24
#define vm7 v28
#define vm8 v32
#define vm9 v36
#define vm10 v40
#define vm11 v44
#define vm12 v48
#define vm13 v52
#define vm14 v56
#define vm15 v60

#endif  // CRT_KELVIN_H_
