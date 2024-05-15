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

#ifndef TFLM_OPT_UTIL_H_
#define TFLM_OPT_UTIL_H_

#include <algorithm>
#include <cstdint>

#define LEFT_SHIFT(_shift) std::max(_shift, 0L)
#define RIGHT_SHIFT(_shift) -std::min(_shift, 0L)

// Use this in place of Tensorflow's
// MultiplyByQuantizedMultiplierSmallerThanOneExp
#define rescale_internal(Vd, Vs, mult, shift, offset, m) \
  do {                                                             \
    vdmulh_w_r_vx##m(Vd, Vs, mult);                                \
    vsha_w_r_vx##m(Vd, Vd, -shift);                                \
    vadd_w_vx##m(Vd, Vd, offset);                                  \
  } while (0);

#define rescale(Vd, Vs, mult, shift, offset) \
  rescale_internal(Vd, Vs, mult, shift,      \
                   offset, );  // NOLINT(whitespace/parens)
#define rescale_m(Vd, Vs, mult, shift, offset) \
  rescale_internal(Vd, Vs, mult, shift, offset, _m);

#endif  // TFLM_OPT_UTIL_H_
