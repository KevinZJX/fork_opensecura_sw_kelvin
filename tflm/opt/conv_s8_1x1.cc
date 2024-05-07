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

// Convolution based on Kelvin ops
// Data types: input: s8, filter: s8, bias: s32
// Special case for 1x1 filter

#include "tflm/opt/conv_util.h"

namespace kelvin::opt {

void ConvS8K1x1(const tflite::ConvParams& params,
                const int32_t* output_multiplier, const int32_t* output_shift,
                const tflite::RuntimeShape& input_shape,
                const int8_t* input_data,
                const tflite::RuntimeShape& filter_shape,
                const int8_t* filter_data,
                const tflite::RuntimeShape& bias_shape,
                const int32_t* bias_data,
                const tflite::RuntimeShape& output_shape, int8_t* output_data) {
  const auto batches = MatchingDim(input_shape, 0, output_shape, 0);
  const auto input_depth = input_shape.Dims(3);
  const auto input_offset = params.input_offset;
  const auto output_height = output_shape.Dims(1);
  const auto output_width = output_shape.Dims(2);
  const auto output_depth = output_shape.Dims(3);
  const auto output_offset = params.output_offset;
  const auto output_activation_min = params.quantized_activation_min;
  const auto output_activation_max = params.quantized_activation_max;
  //  ToDo : support group convolutions.
  int32_t bias[8 * 4];
  int32_t mult[8 * 4];
  int32_t shft[8 * 4];
  union {
    vconv_u8_t conv;
    uint32_t raw;
  } cmds;
  cmds.conv.mode = 0;
  cmds.conv.start = 0;
  cmds.conv.stop = 7;
  cmds.conv.sbias1 = input_offset;
  cmds.conv.sdata1 = true;
  cmds.conv.sbias2 = 0;
  cmds.conv.sdata2 = true;
  for (int zo_hi = 0; zo_hi < output_depth; zo_hi += 8) {
    // transpose filter weigths to support outer prodcut multiplication
    int8_t juggled_filter_data[1][1][1][input_depth / 4][8][4];
    Filter_N_H_W_M<8>(filter_data, juggled_filter_data[0][0][0][0][0], 1, 1,
                      32);

    Swizzle(bias_data, bias, 8);
    Swizzle(output_multiplier, mult, 8);
    Swizzle(output_shift, shft, 8, true);
    int out = 0;
    for (; out + 8 <= output_height * output_width * batches; out += 8) {
      // resetting accumulators to clean up old output
      vdup_b_x_m(v48, 0);
      vdup_b_x_m(v52, 0);

      int in = 0;
      for (; in <= input_depth; in += 32) {
        vld_b_s_xx_m(v0, input_data + out * input_depth + in, input_depth);
        vld_b_s_xx_m(v4, input_data + out * input_depth + in + 4 * input_depth,
                     input_depth);

        vld_b_x_m(v8, juggled_filter_data[0][0][0][in / 32][0][0]);
        vld_b_x_m(v12, juggled_filter_data[0][0][0][(in / 32) + 4][0][0]);

        aconv_vxv(v48, v0, cmds, v8);
      }

      INT32_TO_INT8_OUTPUT_PIPELINE(bias, mult, shft, output_activation_min,
                                    output_activation_max, output_offset, v16,
                                    v20, v24);

      // store the results to ouput memory
      int8_t* p_out = output_data + (out * output_depth) + zo_hi;
      vstq_b_sp_xx(v48, p_out, output_depth);
      vstq_b_sp_xx(v52, p_out, output_depth);
    }
  }
}

}  // namespace kelvin::opt
