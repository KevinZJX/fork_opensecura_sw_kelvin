// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef TFLM_OPT_UTIL_H_
#define TFLM_OPT_UTIL_H_

#include <algorithm>
#include <cstdint>

#define LEFT_SHIFT(_shift) std::max(_shift, 0L)
#define RIGHT_SHIFT(_shift) -std::min(_shift, 0L)

#define rescale_internal(Vd, Vs, mult, shift, offset, m) \
  do {                                                   \
    int32_t _shift = RIGHT_SHIFT(shift);                 \
    vdmulh_w_r_vx##m(Vd, Vs, mult);                      \
    vsha_w_r_vx##m(Vd, Vd, _shift);                      \
    vadd_w_vx##m(Vd, Vd, offset);                        \
  } while (0);

#define rescale(Vd, Vs, mult, shift, offset) rescale_internal(Vd, Vs, mult, shift, offset, );
#define rescale_m(Vd, Vs, mult, shift, offset) rescale_internal(Vd, Vs, mult, shift, offset, _m);

#endif  // TFLM_OPT_UTIL_H_
