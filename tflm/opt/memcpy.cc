// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "crt/kelvin.h"

namespace kelvin::opt {

void *memcpy(void *dst, const void *src, size_t n) {
  const uint8_t *s = reinterpret_cast<const uint8_t *>(src);
  uint8_t *d = reinterpret_cast<uint8_t *>(dst);
  int vl;
  while (true) {
    if (n <= 0) break;
    getvl_b_x_m(vl, n);
    n -= vl;
    vld_b_lp_xx_m(v0, s, vl);
    vst_b_lp_xx_m(v0, d, vl);

    if (n <= 0) break;
    getvl_b_x_m(vl, n);
    n -= vl;
    vld_b_lp_xx_m(v4, s, vl);
    vst_b_lp_xx_m(v4, d, vl);
  }
  return dst;
}

}  // namespace kelvin::opt
