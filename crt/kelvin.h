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

// Stop printing the string when \0 is found in the word.
static inline bool WordHasZero(uint32_t data) {
  return (((data >> 24) & 0xff) == 0) || (((data >> 16) & 0xff) == 0) ||
         (((data >> 8) & 0xff) == 0) || ((data & 0xff) == 0);
}

template <typename T>
static inline void PrintArg(const T arg) {
  if (std::is_same<T, const uint8_t *>::value ||
      std::is_same<T, const char *>::value) {
    klog(arg);
  } else if (std::is_same<T, uint8_t *>::value ||
             std::is_same<T, char *>::value) {
    const uint32_t *p_str = reinterpret_cast<const uint32_t *>(arg);
    uint32_t data = 0;
    do {
      data = *p_str;
      p_str++;
      clog(data);
    } while (!WordHasZero(data));
  } else {  // scalar argument.
    slog(arg);
  }
}

// General printf helper function. The c++11 pack expansion + braced-init-list
// is used to support arbitrary variadic template.
// The unused list is initialized by expanding the arguments in order, and then
// processed by `PrintArg`.
template <typename... Types>
static inline void printf(const char *format, Types... args) {
  constexpr auto size = sizeof...(args);
  if (size > 0) {
    __attribute__((unused)) int x[] = {0, ((void)PrintArg(args), 0)...};
  }
  flog(format);
}

#endif  // CRT_KELVIN_H_
