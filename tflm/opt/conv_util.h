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

inline void Filter_N_H_W_M(const int8_t* input, int8_t* output, int N, int H,
                           int W, int M) {
  const int8_t(&in)[8][H][W][M] = *(int8_t(*)[8][H][W][M])input;
  int8_t(&out)[H][W][M / 4][8][4] = *(int8_t(*)[H][W][M / 4][8][4]) output;
  assert(M >= 4);
  for (int zo = 0; zo < N; ++zo) {
    for (int ky = 0; ky < H; ++ky) {
      for (int kx = 0; kx < W; ++kx) {
        for (int zi = 0; zi < M; ++zi) {
          const int zi_hi = zi >> 2;  // div4
          const int zi_lo = zi & 3;   // rem4
          out[ky][kx][zi_hi][zo][zi_lo] = in[zo][ky][kx][zi];
        }
      }
    }
  }
  // Zero out the rest of the output.
  for (int zo = N; zo < 8; ++zo) {
    for (int ky = 0; ky < H; ++ky) {
      for (int kx = 0; kx < W; ++kx) {
        for (int zi = 0; zi < M; ++zi) {
          const int zi_hi = zi >> 2;  // div4
          const int zi_lo = zi & 3;   // rem4
          out[ky][kx][zi_hi][zo][zi_lo] = 0;
        }
      }
    }
  }
}

// Swizzle values, and duplicate 4 times for stripmining.
inline void Swizzle(const int32_t* input, int32_t* output, int N,
                    bool negate = false) {
  assert(N <= 8);
  const int32_t(&in)[8] = *(int32_t(*)[8])input;
  int32_t(&out)[32] = *(int32_t(*)[32]) output;
  // Convert to accumulator swizzle pattern.
  memset(out, 0, 32 * sizeof(int32_t));
  int offsets[] = {0, 16, 8, 24, 1, 17, 9, 25};
  for (int i = 0; i < N; ++i) {
    int offset = offsets[i];
    out[0 + offset] = out[2 + offset] = out[4 + offset] = out[6 + offset] = in[i];
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

// As above, but interleaves 2 sets of outputs.
#define INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(result0, result1, mult, shft,  \
                                              output_min, \
                                              output_max, output_offset)      \
  {                                                                           \
    vdmulh_w_rn_vv_m(result0, result0, mult);                                   \
    vdmulh_w_rn_vv_m(result1, result1, mult);                                   \
    vsha_w_r_vv_m(result0, result0, shft);                                      \
    vsha_w_r_vv_m(result1, result1, shft);                                      \
    vadd_w_vx_m(result0, result0, output_offset);                               \
    vadd_w_vx_m(result1, result1, output_offset);                               \
    vmax_w_vx_m(result0, result0, output_activation_min);                       \
    vmax_w_vx_m(result1, result1, output_activation_min);                       \
    vmin_w_vx_m(result0, result0, output_activation_max);                       \
    vmin_w_vx_m(result1, result1, output_activation_max);                       \
  }

// As above, but interleaves 3 sets of outputs.
#define INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE3(result0, result1, result2, \
                                              mult, shft,  \
                                              output_min, \
                                              output_max, output_offset)      \
  {                                                                           \
    vdmulh_w_rn_vv_m(result0, result0, mult);                                   \
    vdmulh_w_rn_vv_m(result1, result1, mult);                                   \
    vdmulh_w_rn_vv_m(result2, result2, mult);                                   \
    vsha_w_r_vv_m(result0, result0, shft);                                      \
    vsha_w_r_vv_m(result1, result1, shft);                                      \
    vsha_w_r_vv_m(result2, result2, shft);                                      \
    vadd_w_vx_m(result0, result0, output_offset);                               \
    vadd_w_vx_m(result1, result1, output_offset);                               \
    vadd_w_vx_m(result2, result2, output_offset);                               \
    vmax_w_vx_m(result0, result0, output_activation_min);                       \
    vmax_w_vx_m(result1, result1, output_activation_min);                       \
    vmax_w_vx_m(result2, result2, output_activation_min);                       \
    vmin_w_vx_m(result0, result0, output_activation_max);                       \
    vmin_w_vx_m(result1, result1, output_activation_max);                       \
    vmin_w_vx_m(result2, result2, output_activation_max);                       \
  }

// As above, but interleaves 4 sets of outputs.
#define INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE4(result0, result1, result2, result3, mult, shft,  \
                                              output_min, \
                                              output_max, output_offset)      \
  {                                                                           \
    vdmulh_w_rn_vv_m(result0, result0, mult);                                   \
    vdmulh_w_rn_vv_m(result1, result1, mult);                                   \
    vdmulh_w_rn_vv_m(result2, result2, mult);                                   \
    vdmulh_w_rn_vv_m(result3, result3, mult);                                   \
    vsha_w_r_vv_m(result0, result0, shft);                                      \
    vsha_w_r_vv_m(result1, result1, shft);                                      \
    vsha_w_r_vv_m(result2, result2, shft);                                      \
    vsha_w_r_vv_m(result3, result3, shft);                                      \
    vadd_w_vx_m(result0, result0, output_offset);                               \
    vadd_w_vx_m(result1, result1, output_offset);                               \
    vadd_w_vx_m(result2, result2, output_offset);                               \
    vadd_w_vx_m(result3, result3, output_offset);                               \
    vmax_w_vx_m(result0, result0, output_activation_min);                       \
    vmax_w_vx_m(result1, result1, output_activation_min);                       \
    vmax_w_vx_m(result2, result2, output_activation_min);                       \
    vmax_w_vx_m(result3, result3, output_activation_min);                       \
    vmin_w_vx_m(result0, result0, output_activation_max);                       \
    vmin_w_vx_m(result1, result1, output_activation_max);                       \
    vmin_w_vx_m(result2, result2, output_activation_max);                       \
    vmin_w_vx_m(result3, result3, output_activation_max);                       \
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
