/*
 * Copyright 2024 Google LLC
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

#include "crt/kelvin.h"

namespace kelvin::opt {

void *Memcpy(void *dst, const void *src, size_t n) {
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
