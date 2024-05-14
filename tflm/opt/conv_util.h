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

#ifndef TFLM_OPT_CONV_UTIL_H_
#define TFLM_OPT_CONV_UTIL_H_

#include <cassert>
#include <memory>

#include "crt/kelvin.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/kernels/internal/runtime_shape.h"
#include "tensorflow/lite/kernels/internal/types.h"
#include "tflm/opt/util.h"

namespace kelvin::opt {
/* clang-format off */
constexpr const int swizzle[16] = {
    0, 4, 8, 12,
    2, 6, 10, 14,
    1, 5, 9, 13,
    3, 7, 11, 15,
};
/* clang-format on */

constexpr int kFilterHeightIndex = 1;
constexpr int kFilterWidthIndex = 2;
constexpr int kFilterInputChannelIndex = 3;
constexpr int kInputChannelIndex = 3;
constexpr int kOutputChannelIndex = 3;

#define INA0 v0
#define FLTA0 v8
#define FLTA1 v9
#define FLTA2 v10
#define FLTA3 v11
#define FLTA4 v12
#define FLTA5 v13
#define FLTA6 v14
#define FLTA7 v15
#define ACC v48
#define ACC0 v48
#define OUT0 v56

// H,W ( height and width of filter) N -number of inputs, M -number of outputs
template <int N>
inline void Filter_N_H_W_M(const int8_t* input, int8_t* output, int H, int W,
                           int M) {
  // Convert: input  [zo][ky][kx][zi] (N,3,1,M)
  //          output [zo.hi=N/8][ky][kx][zi_hi=M/4][zo.lo=8][zi_lo=4]
  const int8_t(&in)[N][H][W][M] = *(int8_t(*)[N][H][W][M])input;
  int8_t(&out)[N / 8][H][W][M / 4][8][4] =
      *(int8_t(*)[N / 8][H][W][M / 4][8][4]) output;
  assert(N >= 4 && M >= 4);
  for (int zo = 0; zo < N; ++zo) {
    for (int ky = 0; ky < H; ++ky) {
      for (int kx = 0; kx < W; ++kx) {
        for (int zi = 0; zi < M; ++zi) {
          const int zo_hi = zo >> 3;  // div8
          const int zo_lo = zo & 7;   // rem8
          const int zi_hi = zi >> 2;  // div4
          const int zi_lo = zi & 3;   // rem4
          out[zo_hi][ky][kx][zi_hi][zo_lo][zi_lo] = in[zo][ky][kx][zi];
        }
      }
    }
  }
}

// Swizzle values, and duplicate 4 times for stripmining.
inline void Swizzle(const int32_t* input, int32_t* output, int N,
                    bool negate = false) {
  const int32_t(&in)[N] = *(int32_t(*)[N])input;
  int32_t(&out)[N * 4] = *(int32_t(*)[N * 4]) output;
  // Convert to accumulator swizzle pattern.
  for (int i = 0; i < N / 8; ++i) {
    int32_t* out0 = out + i * 32 + 0;
    int32_t* out1 = out + i * 32 + 16;
    int32_t* out2 = out + i * 32 + 8;
    int32_t* out3 = out + i * 32 + 24;
    for (int j = 0; j < 4; ++j) {
      const int32_t* p_in = in + i * 8;
      for (int k = 0; k < 2; ++k) {
        *out0++ = *p_in++;
        *out1++ = *p_in++;
        *out2++ = *p_in++;
        *out3++ = *p_in++;
      }
    }
  }
  if (negate) {
    for (int i = 0; i < N * 4; ++i) {
      out[i] = -out[i];
    }
  }
}

// Runs strip-mined output pipeline (without bias addition) in place on
// registers.
#define INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(result, mult, shft, output_min, \
                                              output_max, output_offset)      \
  {                                                                           \
    vdmulh_w_rn_vv_m(result, result, mult);                                   \
    vsha_w_r_vv_m(result, result, shft);                                      \
    vadd_w_vx_m(result, result, output_offset);                               \
    vmax_w_vx_m(result, result, output_activation_min);                       \
    vmin_w_vx_m(result, result, output_activation_max);                       \
  }

// Run output pipeline on int32 accumulators in [v48-v55] and store results
// in v48 and v52. Clobbers [v48-v55].
#define INT32_TO_INT8_OUTPUT_PIPELINE(bias, mult, shft, output_min,        \
                                      output_max, output_offset, bias_reg, \
                                      mult_reg, shift_reg)                 \
  {                                                                        \
    vcget(v48);                                                            \
    vld_w_x_m(bias_reg, bias);                                             \
    vld_w_x_m(mult_reg, mult);                                             \
    vld_w_x_m(shift_reg, shft);                                            \
    vadd_w_vv_m(v48, v48, bias_reg);                                       \
    vadd_w_vv_m(v52, v52, bias_reg);                                       \
    vmin_w_vx_m(v48, v48, output_max);                                     \
    vmax_w_vx_m(v52, v52, output_min);                                     \
    vdmulh_w_r_vv_m(v48, v48, mult_reg);                                   \
    vdmulh_w_r_vv_m(v52, v52, mult_reg);                                   \
    vsha_w_r_vv_m(v48, v48, shift_reg);                                    \
    vsha_w_r_vv_m(v52, v52, shift_reg);                                    \
    vadd_w_vx_m(v48, v48, output_offset);                                  \
    vadd_w_vx_m(v52, v52, output_offset);                                  \
    vsraqs_b_vx(v48, v48, 0);                                              \
    vsraqs_b_vx(v52, v52, 0);                                              \
  }
}  // namespace kelvin::opt

#endif  // TFLM_OPT_CONV_UTIL_H_
