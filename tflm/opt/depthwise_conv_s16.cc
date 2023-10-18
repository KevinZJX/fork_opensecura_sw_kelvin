/*
 * Copyright 2023 Google LLC
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

#include <algorithm>

#include "crt/kelvin.h"
#include "tflm/opt/opt.h"
#include "tensorflow/lite/kernels/internal/common.h"

namespace kelvin::opt {

void DepthwiseConv2DKelvinS16K3x1(const int16_t* activations,
                                  const int8_t* weights,
                                  const int64_t* biases,
                                  int channels, int frames, int dilation,
                                  const int32_t* output_mult,
                                  const int32_t* output_shift,
                                  int32_t output_activation_min,
                                  int32_t output_activation_max,
                                  int16_t* output) {
  for (int c = 0; c + 32 <= channels; c += 32) {
    // Load weights and interleave into correct order [v58-v63].
    // Because there are more activations than weights, interleave weights.
    const int8_t* local_weights0 = weights + c;
    vld_b_p_xx(v0, local_weights0, channels);
    vaddw_h_vx(v48, v0, 0);
    vzip_h_vv(v58, v48, v49);

    vld_b_p_xx(v1, local_weights0, channels);
    vaddw_h_vx(v50, v1, 0);
    vzip_h_vv(v60, v50, v51);

    vld_b_x(v2, local_weights0);
    vaddw_h_vx(v52, v2, 0);
    vzip_h_vv(v62, v52, v53);

    // Assume biases fit in 32-bit. This assumption is verified offline.
    // Load biases and swizzle [v52-v55].
    int32_t local_biases[32];
    for (int j = 0; j < 32; j++) {
      local_biases[j] = static_cast<int32_t>(biases[c + j]);
    }
    vld_w_x_m(v4, local_biases);
    vzip_w_vv(v52, v4, v5);
    vzip_w_vv(v54, v6, v7);

    const int32_t step = dilation * channels;
    const int32_t* local_output_mult = output_mult + c;
    const int32_t* local_output_shift = output_shift + c;
    for (int d = 0; d < dilation; d++) {
      // Accumulators will be [v48 - v51].
      const int16_t* local_activations0 = activations + (d * channels) + c;
      const int16_t* local_activations1 = local_activations0 + 16;
      int16_t* local_output = output + (d * channels) + c;

      // Registers [v0-v5 will be for loading activations]
      // Preload for valid padding:
      vld_h_p_xx(v0, local_activations0, step);
      vld_h_p_xx(v1, local_activations1, step);
      vld_h_p_xx(v2, local_activations0, step);
      vld_h_p_xx(v3, local_activations1, step);

      int frames_idx = (2 * dilation) + d;
      int32_t accumulators[32];
      for (; frames_idx < frames; frames_idx += dilation) {
        vld_h_p_xx(v4, local_activations0, step);
        vld_h_p_xx(v5, local_activations1, step);
        vmulw_w_vv(v48, v58, v0);  // Clobber accumulator
        vmulw_w_vv(v50, v59, v1);  // Clobber accumulator
        vadd_w_vv_m(v48, v48, v52);  // Add bias.
        vmulw_w_vv(v40, v60, v2);
        vmulw_w_vv(v42, v61, v3);
        vadd_w_vv_m(v48, v48, v40);
        vmulw_w_vv(v44, v62, v4);
        vmulw_w_vv(v46, v63, v5);
        vadd_w_vv_m(v48, v48, v44);

        vzip_w_vv(v48, v48, v49);  // Swizzle accumulators
        vzip_w_vv(v50, v50, v51);

        vst_w_x_m(v48, accumulators);  // Store accumulators

        // Output pipeline in scalar, to preserve bit accuracy with the ARM CPU
        // implementation.
        for (int i = 0; i < 32; i++) {
          int32_t result = tflite::MultiplyByQuantizedMultiplier(
              static_cast<int64_t>(accumulators[i]), local_output_mult[i],
              local_output_shift[i]);

          local_output[i] = static_cast<int16_t>(
              std::clamp(result, output_activation_min, output_activation_max));
        }

        // Slide registers
        vmvp_vv(v0, v2, v3);
        vmvp_vv(v2, v4, v5);

        local_output += step;
      }
    }
  }
  // TODO(derekjchow): Handle channels % 32 cases.
  // Break it down into:
  //   - one loop looking for 16 byte stripes
  //   - one final loop handling remainder
}

}  // namespace kelvin::opt
