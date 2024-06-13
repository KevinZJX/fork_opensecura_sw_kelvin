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

#include <algorithm>

#include "tflm/opt/conv_util.h"

namespace kelvin::opt {
namespace {
void JumptableSwizzle(const int32_t* input, int32_t* output, int n) {
  switch (n) {
    case 32:
      output[7] = input[28];
      output[15] = input[30];
      output[23] = input[29];
      output[31] = input[31];
    case 28:
      output[6] = input[24];
      output[14] = input[26];
      output[22] = input[25];
      output[30] = input[27];
    case 24:
      output[5] = input[20];
      output[13] = input[22];
      output[21] = input[21];
      output[29] = input[23];
    case 20:
      output[4] = input[16];
      output[12] = input[18];
      output[20] = input[17];
      output[28] = input[19];
    case 16:
      output[27] = input[15];
      output[19] = input[13];
      output[11] = input[14];
      output[3] = input[12];
    case 12:
      output[2] = input[8];
      output[10] = input[10];
      output[18] = input[9];
      output[26] = input[11];
    case 8:
      output[1] = input[4];
      output[9] = input[6];
      output[17] = input[5];
      output[25] = input[7];
    case 4:
      output[0] = input[0];
      output[8] = input[2];
      output[16] = input[1];
      output[24] = input[3];
  }
}

}  // namespace

#define FLT_0_0 v0
#define FLT_0_1 v3
#define FLT_0_2 v6
#define FLT_0_3 v9
#define FLT_0_4 v12

#define FLT_1_0 v1
#define FLT_1_1 v4
#define FLT_1_2 v7
#define FLT_1_3 v10
#define FLT_1_4 v13

#define FLT_2_0 v2
#define FLT_2_1 v5
#define FLT_2_2 v8
#define FLT_2_3 v11
#define FLT_2_4 v14

#define FLT_3_0 v15
#define FLT_3_1 v16
#define FLT_3_2 v17
#define FLT_3_3 v18
#define FLT_3_4 v19

#define FLT_HOLE v20
#define FLT_4_0 v21
#define FLT_4_1 v22
#define FLT_4_2 v23
#define FLT_4_3 v24
#define FLT_4_4 v25

#define INPUT_0_0 v26
#define INPUT_0_1 v29
#define INPUT_0_2 v32
#define INPUT_0_3 v35
#define INPUT_0_4 v38

#define INPUT_1_0 v27
#define INPUT_1_1 v30
#define INPUT_1_2 v33
#define INPUT_1_3 v36
#define INPUT_1_4 v39

#define INPUT_2_0 v28
#define INPUT_2_1 v31
#define INPUT_2_2 v34
#define INPUT_2_3 v37
#define INPUT_2_4 v40

#define INPUT_3_0 v41
#define INPUT_3_1 v42
#define INPUT_3_2 v43
#define INPUT_3_3 v44
#define INPUT_3_4 v45

#define INPUT_4_0 v46
#define INPUT_4_1 v47
#define INPUT_4_2 v48
#define INPUT_4_3 v49
#define INPUT_4_4 v50

#define CALCULATE_IN_X(in_x_origin)                        \
  {                                                        \
    _Pragma("GCC unroll 5") for (int i = 0; i < 5; ++i) {  \
      in_x[i] = in_x_origin + (dilation_width_factor * i); \
    }                                                      \
  }

#define CALCULATE_IN_Y(in_y_origin)                         \
  {                                                         \
    _Pragma("GCC unroll 5") for (int i = 0; i < 5; ++i) {   \
      in_y[i] = in_y_origin + (dilation_height_factor * i); \
    }                                                       \
  }

#define PAD_ROW_0(input_offset)   \
  {                               \
    vdup_b_x(INPUT_0_0, -input_offset); \
    vdup_b_x(INPUT_0_1, -input_offset); \
    vdup_b_x(INPUT_0_2, -input_offset); \
    vdup_b_x(INPUT_0_3, -input_offset); \
    vdup_b_x(INPUT_0_4, -input_offset); \
  }
#define PAD_ROW_1(input_offset)   \
  {                               \
    vdup_b_x(INPUT_1_0, -input_offset); \
    vdup_b_x(INPUT_1_1, -input_offset); \
    vdup_b_x(INPUT_1_2, -input_offset); \
    vdup_b_x(INPUT_1_3, -input_offset); \
    vdup_b_x(INPUT_1_4, -input_offset); \
  }
#define PAD_ROW_2(input_offset)   \
  {                            \
    vdup_b_x(INPUT_2_0, -input_offset); \
    vdup_b_x(INPUT_2_1, -input_offset); \
    vdup_b_x(INPUT_2_2, -input_offset); \
    vdup_b_x(INPUT_2_3, -input_offset); \
    vdup_b_x(INPUT_2_4, -input_offset); \
  }
#define PAD_ROW_3(input_offset)   \
  {                               \
    vdup_b_x(INPUT_3_0, -input_offset); \
    vdup_b_x(INPUT_3_1, -input_offset); \
    vdup_b_x(INPUT_3_2, -input_offset); \
    vdup_b_x(INPUT_3_3, -input_offset); \
    vdup_b_x(INPUT_3_4, -input_offset); \
  }
#define PAD_ROW_4(input_offset)   \
  {                               \
    vdup_b_x(INPUT_4_0, -input_offset); \
    vdup_b_x(INPUT_4_1, -input_offset); \
    vdup_b_x(INPUT_4_2, -input_offset); \
    vdup_b_x(INPUT_4_3, -input_offset); \
    vdup_b_x(INPUT_4_4, -input_offset); \
  }

#define LOAD_ROW_0(p_input, input_width, in_y, in_x)         \
  {                                                          \
    const int8_t* p_row = p_input + (in_y[0] * input_width); \
    vdup_b_x(INPUT_0_0, *(p_row + in_x[0]));                       \
    vdup_b_x(INPUT_0_1, *(p_row + in_x[1]));                       \
    vdup_b_x(INPUT_0_2, *(p_row + in_x[2]));                       \
    vdup_b_x(INPUT_0_3, *(p_row + in_x[3]));                       \
    vdup_b_x(INPUT_0_4, *(p_row + in_x[4]));                       \
  }

#define LOAD_ROW_1(p_input, input_width, in_y, in_x)         \
  {                                                          \
    const int8_t* p_row = p_input + (in_y[1] * input_width); \
    vdup_b_x(INPUT_1_0, *(p_row + in_x[0]));                       \
    vdup_b_x(INPUT_1_1, *(p_row + in_x[1]));                       \
    vdup_b_x(INPUT_1_2, *(p_row + in_x[2]));                       \
    vdup_b_x(INPUT_1_3, *(p_row + in_x[3]));                       \
    vdup_b_x(INPUT_1_4, *(p_row + in_x[4]));                       \
  }

#define LOAD_ROW_2(p_input, input_width, in_y, in_x)         \
  {                                                          \
    const int8_t* p_row = p_input + (in_y[2] * input_width); \
    vdup_b_x(INPUT_2_0, *(p_row + in_x[0]));                       \
    vdup_b_x(INPUT_2_1, *(p_row + in_x[1]));                       \
    vdup_b_x(INPUT_2_2, *(p_row + in_x[2]));                       \
    vdup_b_x(INPUT_2_3, *(p_row + in_x[3]));                       \
    vdup_b_x(INPUT_2_4, *(p_row + in_x[4]));                       \
  }

#define LOAD_ROW_3(p_input, input_width, in_y, in_x)         \
  {                                                          \
    const int8_t* p_row = p_input + (in_y[3] * input_width); \
    vdup_b_x(INPUT_3_0, *(p_row + in_x[0]));                       \
    vdup_b_x(INPUT_3_1, *(p_row + in_x[1]));                       \
    vdup_b_x(INPUT_3_2, *(p_row + in_x[2]));                       \
    vdup_b_x(INPUT_3_3, *(p_row + in_x[3]));                       \
    vdup_b_x(INPUT_3_4, *(p_row + in_x[4]));                       \
  }

#define LOAD_ROW_4(p_input, input_width, in_y, in_x)         \
  {                                                          \
    const int8_t* p_row = p_input + (in_y[4] * input_width); \
    vdup_b_x(INPUT_4_0, *(p_row + in_x[0]));                       \
    vdup_b_x(INPUT_4_1, *(p_row + in_x[1]));                       \
    vdup_b_x(INPUT_4_2, *(p_row + in_x[2]));                       \
    vdup_b_x(INPUT_4_3, *(p_row + in_x[3]));                       \
    vdup_b_x(INPUT_4_4, *(p_row + in_x[4]));                       \
  }

#define H_PAD_OR_LOAD_ROW_0(p_input, input_width, input_offset, in_y, in_x) \
  if (in_x[0] >= 0 && in_x[4] < input_width) {                              \
    LOAD_ROW_0(p_input, input_width, in_y, in_x);                           \
  } else {                                                                  \
    const int8_t* p_row = p_input + (in_y[0] * input_width);                \
    if (in_x[0] < 0 || in_x[0] >= input_width) {                            \
      vdup_b_x(INPUT_0_0, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_0_0, *(p_row + in_x[0]));                                    \
    }                                                                       \
    if (in_x[1] < 0 || in_x[1] >= input_width) {                            \
      vdup_b_x(INPUT_0_1, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_0_1, *(p_row + in_x[1]));                                    \
    }                                                                       \
    if (in_x[2] < 0 || in_x[2] >= input_width) {                            \
      vdup_b_x(INPUT_0_2, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_0_2, *(p_row + in_x[2]));                                    \
    }                                                                       \
    if (in_x[3] < 0 || in_x[3] >= input_width) {                            \
      vdup_b_x(INPUT_0_3, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_0_3, *(p_row + in_x[3]));                                    \
    }                                                                       \
    if (in_x[4] < 0 || in_x[4] >= input_width) {                            \
      vdup_b_x(INPUT_0_4, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_0_4, *(p_row + in_x[4]));                                    \
    }                                                                       \
  }

#define H_PAD_OR_LOAD_ROW_1(p_input, input_width, input_offset, in_y, in_x) \
  if (in_x[0] >= 0 && in_x[4] < input_width) {                              \
    LOAD_ROW_1(p_input, input_width, in_y, in_x);                           \
  } else {                                                                  \
    const int8_t* p_row = p_input + (in_y[1] * input_width);                \
    if (in_x[0] < 0 || in_x[0] >= input_width) {                            \
      vdup_b_x(INPUT_1_0, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_1_0, *(p_row + in_x[0]));                                    \
    }                                                                       \
    if (in_x[1] < 0 || in_x[1] >= input_width) {                            \
      vdup_b_x(INPUT_1_1, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_1_1, *(p_row + in_x[1]));                                    \
    }                                                                       \
    if (in_x[2] < 0 || in_x[2] >= input_width) {                            \
      vdup_b_x(INPUT_1_2, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_1_2, *(p_row + in_x[2]));                                    \
    }                                                                       \
    if (in_x[3] < 0 || in_x[3] >= input_width) {                            \
      vdup_b_x(INPUT_1_3, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_1_3, *(p_row + in_x[3]));                                    \
    }                                                                       \
    if (in_x[4] < 0 || in_x[4] >= input_width) {                            \
      vdup_b_x(INPUT_1_4, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_1_4, *(p_row + in_x[4]));                                    \
    }                                                                       \
  }

#define H_PAD_OR_LOAD_ROW_2(p_input, input_width, input_offset, in_y, in_x) \
  if (in_x[0] >= 0 && in_x[4] < input_width) {                              \
    LOAD_ROW_2(p_input, input_width, in_y, in_x);                           \
  } else {                                                                  \
    const int8_t* p_row = p_input + (in_y[2] * input_width);                \
    if (in_x[0] < 0 || in_x[0] >= input_width) {                            \
      vdup_b_x(INPUT_2_0, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_2_0, *(p_row + in_x[0]));                                    \
    }                                                                       \
    if (in_x[1] < 0 || in_x[1] >= input_width) {                            \
      vdup_b_x(INPUT_2_1, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_2_1, *(p_row + in_x[1]));                                    \
    }                                                                       \
    if (in_x[2] < 0 || in_x[2] >= input_width) {                            \
      vdup_b_x(INPUT_2_2, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_2_2, *(p_row + in_x[2]));                                    \
    }                                                                       \
    if (in_x[3] < 0 || in_x[3] >= input_width) {                            \
      vdup_b_x(INPUT_2_3, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_2_3, *(p_row + in_x[3]));                                    \
    }                                                                       \
    if (in_x[4] < 0 || in_x[4] >= input_width) {                            \
      vdup_b_x(INPUT_2_4, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_2_4, *(p_row + in_x[4]));                                    \
    }                                                                       \
  }

#define H_PAD_OR_LOAD_ROW_3(p_input, input_width, input_offset, in_y, in_x) \
  if (in_x[0] >= 0 && in_x[4] < input_width) {                              \
    LOAD_ROW_3(p_input, input_width, in_y, in_x);                           \
  } else {                                                                  \
    const int8_t* p_row = p_input + (in_y[3] * input_width);                \
    if (in_x[0] < 0 || in_x[0] >= input_width) {                            \
      vdup_b_x(INPUT_3_0, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_3_0, *(p_row + in_x[0]));                                    \
    }                                                                       \
    if (in_x[1] < 0 || in_x[1] >= input_width) {                            \
      vdup_b_x(INPUT_3_1, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_3_1, *(p_row + in_x[1]));                                    \
    }                                                                       \
    if (in_x[2] < 0 || in_x[2] >= input_width) {                            \
      vdup_b_x(INPUT_3_2, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_3_2, *(p_row + in_x[2]));                                    \
    }                                                                       \
    if (in_x[3] < 0 || in_x[3] >= input_width) {                            \
      vdup_b_x(INPUT_3_3, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_3_3, *(p_row + in_x[3]));                                    \
    }                                                                       \
    if (in_x[4] < 0 || in_x[4] >= input_width) {                            \
      vdup_b_x(INPUT_3_4, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_3_4, *(p_row + in_x[4]));                                    \
    }                                                                       \
  }

#define H_PAD_OR_LOAD_ROW_4(p_input, input_width, input_offset, in_y, in_x) \
  if (in_x[0] >= 0 && in_x[4] < input_width) {                              \
    LOAD_ROW_4(p_input, input_width, in_y, in_x);                           \
  } else {                                                                  \
    const int8_t* p_row = p_input + (in_y[4] * input_width);                \
    if (in_x[0] < 0 || in_x[0] >= input_width) {                            \
      vdup_b_x(INPUT_4_0, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_4_0, *(p_row + in_x[0]));                                    \
    }                                                                       \
    if (in_x[1] < 0 || in_x[1] >= input_width) {                            \
      vdup_b_x(INPUT_4_1, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_4_1, *(p_row + in_x[1]));                                    \
    }                                                                       \
    if (in_x[2] < 0 || in_x[2] >= input_width) {                            \
      vdup_b_x(INPUT_4_2, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_4_2, *(p_row + in_x[2]));                                    \
    }                                                                       \
    if (in_x[3] < 0 || in_x[3] >= input_width) {                            \
      vdup_b_x(INPUT_4_3, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_4_3, *(p_row + in_x[3]));                                    \
    }                                                                       \
    if (in_x[4] < 0 || in_x[4] >= input_width) {                            \
      vdup_b_x(INPUT_4_4, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(INPUT_4_4, *(p_row + in_x[4]));                                    \
    }                                                                       \
  }

#define _H_PAD_OR_LOAD_ROW(row, p_input, input_width, input_offset, in_y, \
                           in_x)                                          \
  H_PAD_OR_LOAD_ROW_##row(p_input, input_width, input_offset, in_y, in_x);

#define _PAD_OR_LOAD_ROW(row, p_input, input_height, input_width, in_y, in_x,  \
                         input_offset)                                         \
  {                                                                            \
    if (in_y[row] < 0 || in_y[row] >= input_height) {                          \
      PAD_ROW_##row(input_offset);                                             \
    } else {                                                                   \
      _H_PAD_OR_LOAD_ROW(row, p_input, input_width, input_offset, in_y, in_x); \
    }                                                                          \
  }

#define PAD_OR_LOAD_ROW_0(p_input, input_height, input_width, in_y, in_x, \
                          input_offset)                                   \
  _PAD_OR_LOAD_ROW(0, p_input, input_height, input_width, in_y, in_x,     \
                   input_offset);
#define PAD_OR_LOAD_ROW_1(p_input, input_height, input_width, in_y, in_x, \
                          input_offset)                                   \
  _PAD_OR_LOAD_ROW(1, p_input, input_height, input_width, in_y, in_x,     \
                   input_offset);
#define PAD_OR_LOAD_ROW_2(p_input, input_height, input_width, in_y, in_x, \
                          input_offset)                                   \
  _PAD_OR_LOAD_ROW(2, p_input, input_height, input_width, in_y, in_x,     \
                   input_offset);
#define PAD_OR_LOAD_ROW_3(p_input, input_height, input_width, in_y, in_x, \
                          input_offset)                                   \
  _PAD_OR_LOAD_ROW(3, p_input, input_height, input_width, in_y, in_x,     \
                   input_offset);
#define PAD_OR_LOAD_ROW_4(p_input, input_height, input_width, in_y, in_x, \
                          input_offset)                                   \
  _PAD_OR_LOAD_ROW(4, p_input, input_height, input_width, in_y, in_x,     \
                   input_offset);

#define COMPUTE(cmds, swizzled_bias_data) \
  {                                       \
    vld_w_x_m(v60, swizzled_bias_data);   \
    adwinit_v(v60, v60);                  \
    adwconv_vxv(v60, INPUT_0_0, cmds, FLT_0_0);      \
    adwconv_vxv(v60, INPUT_0_1, cmds, FLT_0_1);      \
    adwconv_vxv(v60, INPUT_0_2, cmds, FLT_0_2);      \
    adwconv_vxv(v60, INPUT_0_3, cmds, FLT_0_3);      \
    adwconv_vxv(v60, INPUT_0_4, cmds, FLT_0_4);     \
    adwconv_vxv(v60, INPUT_3_0, cmds, FLT_3_0);     \
    adwconv_vxv(v60, INPUT_3_3, cmds, FLT_3_3);     \
    adwconv_vxv(v60, INPUT_3_4, cmds, FLT_HOLE);     \
    vdwconv_vxv(v60, INPUT_4_2, cmds, FLT_4_2);     \
  }

#define OUTPUT(output_activation_min, output_activation_max, output_offset, \
               local_output_data, n_channels)                               \
  {                                                                         \
    INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(                                  \
        v60, v52, v56, output_activation_min, output_activation_max,        \
        output_offset);                                                     \
    vsraqs_b_vx(v60, v60, 0);                                               \
    vst_b_l_xx(v60, local_output_data, n_channels);                         \
  }

// Estimated count of arithmetic ops: 58.297 M  ops, equivalently 29.148 M  MACs
void ConvPerChannelD1OD24_5x5(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  // const int batches = tflite::MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth =
      tflite::MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_NE(groups, 0);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  TFLITE_DCHECK_NE(filters_per_group, 0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  // Scratch pads to juggle data
  const size_t swizzled_filter_data_size = 24 * filter_height * filter_width;
  std::unique_ptr<int8_t> swizzled_filter_data(reinterpret_cast<int8_t*>(
      ::aligned_alloc(32, swizzled_filter_data_size)));
  int32_t swizzled_bias_data[32];
  int32_t swizzled_output_multiplier[32];
  int32_t swizzled_output_shift[32];
  // Transpose filter for easy loading
  for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
    for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
      for (int i = 0; i < 24; i++) {
        int filter_location =
            (filter_y * filter_width * 24) + (filter_x * 24) + i;
        swizzled_filter_data.get()[filter_location] =
            filter_data[tflite::Offset(filter_shape, i, filter_y, filter_x, 0)];
      }
    }
  }

  const int8_t* p_flt_0 = swizzled_filter_data.get() + (0 * filter_width * 24);
  const int8_t* p_flt_1 = swizzled_filter_data.get() + (1 * filter_width * 24);
  const int8_t* p_flt_2 = swizzled_filter_data.get() + (2 * filter_width * 24);
  const int8_t* p_flt_3 = swizzled_filter_data.get() + (3 * filter_width * 24);
  const int8_t* p_flt_4 = swizzled_filter_data.get() + (4 * filter_width * 24);
  vld_b_lp_xx(FLT_0_0, p_flt_0, 24);
  vld_b_lp_xx(FLT_0_1, p_flt_0, 24);
  vld_b_lp_xx(FLT_0_2, p_flt_0, 24);
  vld_b_lp_xx(FLT_0_3, p_flt_0, 24);
  vld_b_lp_xx(FLT_0_4, p_flt_0, 24);

  vld_b_lp_xx(FLT_1_0, p_flt_1, 24);
  vld_b_lp_xx(FLT_1_1, p_flt_1, 24);
  vld_b_lp_xx(FLT_1_2, p_flt_1, 24);
  vld_b_lp_xx(FLT_1_3, p_flt_1, 24);
  vld_b_lp_xx(FLT_1_4, p_flt_1, 24);

  vld_b_lp_xx(FLT_2_0, p_flt_2, 24);
  vld_b_lp_xx(FLT_2_1, p_flt_2, 24);
  vld_b_lp_xx(FLT_2_2, p_flt_2, 24);
  vld_b_lp_xx(FLT_2_3, p_flt_2, 24);
  vld_b_lp_xx(FLT_2_4, p_flt_2, 24);

  vld_b_lp_xx(FLT_3_0, p_flt_3, 24);
  vld_b_lp_xx(FLT_3_1, p_flt_3, 24);
  vld_b_lp_xx(FLT_3_2, p_flt_3, 24);
  vld_b_lp_xx(FLT_3_3, p_flt_3, 24);
  vld_b_lp_xx(FLT_3_4, p_flt_3, 24);

  vdup_b_x(FLT_HOLE, 0);
  vld_b_lp_xx(FLT_4_0, p_flt_4, 24);
  vld_b_lp_xx(FLT_4_1, p_flt_4, 24);
  vld_b_lp_xx(FLT_4_2, p_flt_4, 24);
  vld_b_lp_xx(FLT_4_3, p_flt_4, 24);
  vld_b_lp_xx(FLT_4_4, p_flt_4, 24);

  union {
    vdwconv_u8_t dwconv;
    uint32_t raw;
  } cmds;
  cmds.raw = 0;
  cmds.dwconv.sdata1 = true;
  cmds.dwconv.sbias1 = input_offset;
  cmds.dwconv.sdata2 = true;
  cmds.dwconv.sbias2 = 0;
  cmds.dwconv.mode = 0;
  cmds.dwconv.sparsity = 0;
  cmds.dwconv.regbase = 0;
  int out_channel = 0;
  int n_channels = 24;

  memset(swizzled_bias_data, 0, 32 * sizeof(uint32_t));
  JumptableSwizzle(bias_data + out_channel, swizzled_bias_data, n_channels);
  memset(swizzled_output_multiplier, 0, 32 * sizeof(uint32_t));
  JumptableSwizzle(output_multiplier + out_channel, swizzled_output_multiplier,
                   n_channels);
  JumptableSwizzle(output_shift + out_channel, swizzled_output_shift,
                   n_channels);
  vld_w_x_m(v52, swizzled_output_multiplier);
  vld_w_x_m(v56, swizzled_output_shift);
  vrsub_w_vx_m(v56, v56, 0);

  int8_t* local_output_data = output_data + out_channel;
  int in_y[5];
  int in_x[7];
  int out_y = 0;
  const int8_t* p_input = input_data;
  // Handle top row padding
  for (; out_y < pad_height; ++out_y) {
    int out_x = 0;
    const int in_y_origin = (out_y * stride_height) - pad_height;
    CALCULATE_IN_Y(in_y_origin);
    // Left padding required
    for (; out_x < pad_width; ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_width;
      CALCULATE_IN_X(in_x_origin);
      PAD_OR_LOAD_ROW_0(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_1(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_2(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_3(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_4(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      COMPUTE(cmds, swizzled_bias_data);
      OUTPUT(output_activation_min, output_activation_max, output_offset,
             local_output_data, n_channels);
      local_output_data += output_depth;
    }
    // No side padding
    for (; out_x < (output_width - pad_width); ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_width;
      CALCULATE_IN_X(in_x_origin);
      PAD_OR_LOAD_ROW_0(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_1(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_2(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_3(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_4(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      COMPUTE(cmds, swizzled_bias_data);
      OUTPUT(output_activation_min, output_activation_max, output_offset,
             local_output_data, n_channels);
      local_output_data += output_depth;
    }
    // Right padding required
    for (; out_x < output_width; ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_width;
      CALCULATE_IN_X(in_x_origin);
      PAD_OR_LOAD_ROW_0(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_1(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_2(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_3(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_4(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      COMPUTE(cmds, swizzled_bias_data);
      OUTPUT(output_activation_min, output_activation_max, output_offset,
             local_output_data, n_channels);
      local_output_data += output_depth;
    }
  }

  // No height padding
  for (; out_y < (output_height - pad_height); ++out_y) {
    const int in_y_origin = (out_y * stride_height) - pad_height;
    CALCULATE_IN_Y(in_y_origin);
    // Left padding
    int out_x = 0;
    for (; out_x < pad_width; ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_width;
      CALCULATE_IN_X(in_x_origin);
      PAD_OR_LOAD_ROW_0(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_1(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_2(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_3(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_4(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      COMPUTE(cmds, swizzled_bias_data);
      OUTPUT(output_activation_min, output_activation_max, output_offset,
             local_output_data, n_channels);
      local_output_data += output_depth;
    }
    for (; out_x + 2 <= (output_width - pad_width); out_x += 2) {
      const int in_x_origin = (out_x * stride_width) - pad_width;

      #pragma GCC unroll 7
      for (int i = 0; i < 7; ++i) {
        in_x[i] = in_x_origin + (dilation_width_factor * i);
      }
      const int8_t* p_rows[5];
      #pragma GCC unroll 5
      for (int i = 0; i < 5; ++i) {
        p_rows[i] = p_input + (in_y[i] * input_width);
      }

      vdup_b_x(INPUT_0_0, *(p_rows[0] + in_x[0]));
      vdup_b_x(INPUT_0_1, *(p_rows[0] + in_x[1]));
      vdup_b_x(INPUT_0_2, *(p_rows[0] + in_x[2]));
      vdup_b_x(INPUT_0_3, *(p_rows[0] + in_x[3]));
      vdup_b_x(INPUT_0_4, *(p_rows[0] + in_x[4]));

      vdup_b_x(INPUT_1_0, *(p_rows[1] + in_x[0]));
      vdup_b_x(INPUT_1_1, *(p_rows[1] + in_x[1]));
      vdup_b_x(INPUT_1_2, *(p_rows[1] + in_x[2]));
      vdup_b_x(INPUT_1_3, *(p_rows[1] + in_x[3]));
      vdup_b_x(INPUT_1_4, *(p_rows[1] + in_x[4]));

      vdup_b_x(INPUT_2_0, *(p_rows[2] + in_x[0]));
      vdup_b_x(INPUT_2_1, *(p_rows[2] + in_x[1]));
      vdup_b_x(INPUT_2_2, *(p_rows[2] + in_x[2]));
      vdup_b_x(INPUT_2_3, *(p_rows[2] + in_x[3]));
      vdup_b_x(INPUT_2_4, *(p_rows[2] + in_x[4]));

      vdup_b_x(INPUT_3_0, *(p_rows[3] + in_x[0]));
      vdup_b_x(INPUT_3_1, *(p_rows[3] + in_x[1]));
      vdup_b_x(INPUT_3_2, *(p_rows[3] + in_x[2]));
      vdup_b_x(INPUT_3_3, *(p_rows[3] + in_x[3]));
      vdup_b_x(INPUT_3_4, *(p_rows[3] + in_x[4]));

      vdup_b_x(INPUT_4_0, *(p_rows[4] + in_x[0]));
      vdup_b_x(INPUT_4_1, *(p_rows[4] + in_x[1]));
      vdup_b_x(INPUT_4_2, *(p_rows[4] + in_x[2]));
      vdup_b_x(INPUT_4_3, *(p_rows[4] + in_x[3]));
      vdup_b_x(INPUT_4_4, *(p_rows[4] + in_x[4]));

      vld_w_x_m(v60, swizzled_bias_data);
      adwinit_v(v60, v60);
      adwconv_vxv(v60, INPUT_0_0, cmds, FLT_0_0);
      adwconv_vxv(v60, INPUT_0_1, cmds, FLT_0_1);
      adwconv_vxv(v60, INPUT_0_2, cmds, FLT_0_2);
      adwconv_vxv(v60, INPUT_0_3, cmds, FLT_0_3);
      adwconv_vxv(v60, INPUT_0_4, cmds, FLT_0_4);
      adwconv_vxv(v60, INPUT_3_0, cmds, FLT_3_0);
      adwconv_vxv(v60, INPUT_3_3, cmds, FLT_3_3);
      adwconv_vxv(v60, INPUT_3_4, cmds, FLT_HOLE);
      vdwconv_vxv(v60, INPUT_4_2, cmds, FLT_4_2);
      vmv_v(INPUT_0_0, v60);
      vmv_v(INPUT_1_0, v61);
      vmv_v(INPUT_2_0, v62);
      vmv_v(INPUT_0_1, v63);

      vdup_b_x(INPUT_3_0, *(p_rows[3] + in_x[5]));
      vdup_b_x(INPUT_3_1, *(p_rows[3] + in_x[6]));

      vmv_v(INPUT_4_0, INPUT_4_2);
      vmv_v(INPUT_4_1, INPUT_4_3);
      vmv_v(INPUT_4_2, INPUT_4_4);
      vdup_b_x(INPUT_4_3, *(p_rows[4] + in_x[5]));
      vdup_b_x(INPUT_4_4, *(p_rows[4] + in_x[6]));

      vld_w_x_m(v60, swizzled_bias_data);
      adwinit_v(v60, v60);
      adwconv_vxv(v60, INPUT_0_2, cmds, FLT_0_0);
      adwconv_vxv(v60, INPUT_0_3, cmds, FLT_0_1);

      vmv_v(INPUT_0_2, INPUT_0_0);
      vmv_v(INPUT_1_2, INPUT_1_0);
      vmv_v(INPUT_2_2, INPUT_2_0);
      vmv_v(INPUT_0_3, INPUT_0_1);

      vdup_b_x(INPUT_0_0, *(p_rows[0] + in_x[5]));
      vdup_b_x(INPUT_0_1, *(p_rows[0] + in_x[6]));
      vdup_b_x(INPUT_1_0, *(p_rows[1] + in_x[5]));
      vdup_b_x(INPUT_1_1, *(p_rows[1] + in_x[6]));
      vdup_b_x(INPUT_2_0, *(p_rows[2] + in_x[5]));
      vdup_b_x(INPUT_2_1, *(p_rows[2] + in_x[6]));

      adwconv_vxv(v60, INPUT_0_4, cmds, FLT_0_2);
      adwconv_vxv(v60, INPUT_0_0, cmds, FLT_0_3);
      adwconv_vxv(v60, INPUT_0_1, cmds, FLT_0_4);
      adwconv_vxv(v60, INPUT_3_2, cmds, FLT_3_0);
      adwconv_vxv(v60, INPUT_3_0, cmds, FLT_3_3);
      adwconv_vxv(v60, INPUT_3_4, cmds, FLT_HOLE);
      vdwconv_vxv(v60, INPUT_4_2, cmds, FLT_4_2);
      INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE2(
        v60, INPUT_0_2, v52, v56, output_activation_min,
        output_activation_max, output_offset
      );
      vsraqs_b_vx(INPUT_0_2, INPUT_0_2, 0);
      vst_b_l_xx(INPUT_0_2, local_output_data, n_channels);
      local_output_data += output_depth;
      vsraqs_b_vx(v60, v60, 0);
      vst_b_l_xx(v60, local_output_data, n_channels);
      local_output_data += output_depth;
    }
    for (; out_x < (output_width - pad_width); ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_width;

      CALCULATE_IN_X(in_x_origin);
      LOAD_ROW_0(p_input, input_width, in_y, in_x);
      LOAD_ROW_1(p_input, input_width, in_y, in_x);
      LOAD_ROW_2(p_input, input_width, in_y, in_x);
      LOAD_ROW_3(p_input, input_width, in_y, in_x);
      LOAD_ROW_4(p_input, input_width, in_y, in_x);
      COMPUTE(cmds, swizzled_bias_data);
      OUTPUT(output_activation_min, output_activation_max, output_offset,
             local_output_data, n_channels);
      local_output_data += output_depth;
    }
    // Right padding
    for (; out_x < output_width; ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_width;
      CALCULATE_IN_X(in_x_origin);
      PAD_OR_LOAD_ROW_0(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_1(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_2(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_3(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_4(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      COMPUTE(cmds, swizzled_bias_data);
      OUTPUT(output_activation_min, output_activation_max, output_offset,
             local_output_data, n_channels);
      local_output_data += output_depth;
    }
  }

  // Handle bottom row padding
  for (; out_y < output_height; ++out_y) {
    const int in_y_origin = (out_y * stride_height) - pad_height;
    CALCULATE_IN_Y(in_y_origin);
    int out_x = 0;
    for (; out_x < pad_width; ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_width;
      CALCULATE_IN_X(in_x_origin);
      PAD_OR_LOAD_ROW_0(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_1(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_2(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_3(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_4(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      COMPUTE(cmds, swizzled_bias_data);
      OUTPUT(output_activation_min, output_activation_max, output_offset,
             local_output_data, n_channels);
      local_output_data += output_depth;
    }
    for (; out_x < (output_width - pad_width); ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_width;
      CALCULATE_IN_X(in_x_origin);
      PAD_OR_LOAD_ROW_0(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_1(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_2(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_3(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_4(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      COMPUTE(cmds, swizzled_bias_data);
      OUTPUT(output_activation_min, output_activation_max, output_offset,
             local_output_data, n_channels);
      local_output_data += output_depth;
    }
    for (; out_x < output_width; ++out_x) {
      const int in_x_origin = (out_x * stride_width) - pad_width;
      CALCULATE_IN_X(in_x_origin);
      PAD_OR_LOAD_ROW_0(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_1(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_2(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_3(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      PAD_OR_LOAD_ROW_4(p_input, input_height, input_width, in_y, in_x,
                        input_offset);
      COMPUTE(cmds, swizzled_bias_data);
      OUTPUT(output_activation_min, output_activation_max, output_offset,
             local_output_data, n_channels);
      local_output_data += output_depth;
    }
  }
}

#undef PAD_OR_LOAD_ROW_0
#undef PAD_OR_LOAD_ROW_1
#undef PAD_OR_LOAD_ROW_2
#undef PAD_OR_LOAD_ROW_3
#undef PAD_OR_LOAD_ROW_4
#undef _PAD_OR_LOAD_ROW
#undef _H_PAD_OR_LOAD_ROW
#undef H_PAD_OR_LOAD_ROW_0
#undef H_PAD_OR_LOAD_ROW_1
#undef H_PAD_OR_LOAD_ROW_2
#undef H_PAD_OR_LOAD_ROW_3
#undef H_PAD_OR_LOAD_ROW_4
#undef LOAD_ROW_0
#undef LOAD_ROW_1
#undef LOAD_ROW_2
#undef LOAD_ROW_3
#undef LOAD_ROW_4
#undef PAD_ROW_0
#undef PAD_ROW_1
#undef PAD_ROW_2
#undef PAD_ROW_3
#undef PAD_ROW_4
#undef CALCULATE_IN_X
#undef CALCULATE_IN_Y
#undef INPUT_0_0
#undef INPUT_0_1
#undef INPUT_0_2
#undef INPUT_0_3
#undef INPUT_0_4
#undef INPUT_1_0
#undef INPUT_1_1
#undef INPUT_1_2
#undef INPUT_1_3
#undef INPUT_1_4
#undef INPUT_2_0
#undef INPUT_2_1
#undef INPUT_2_2
#undef INPUT_2_3
#undef INPUT_2_4
#undef INPUT_3_0
#undef INPUT_3_1
#undef INPUT_3_2
#undef INPUT_3_3
#undef INPUT_3_4
#undef INPUT_4_0
#undef INPUT_4_1
#undef INPUT_4_2
#undef INPUT_4_3
#undef INPUT_4_4
#undef INPUT_0_5
#undef INPUT_1_5
#undef INPUT_2_5
#undef INPUT_3_5
#undef INPUT_4_5
#undef FLT_0_0
#undef FLT_0_1
#undef FLT_0_2
#undef FLT_0_3
#undef FLT_0_4
#undef FLT_1_0
#undef FLT_1_1
#undef FLT_1_2
#undef FLT_1_3
#undef FLT_1_4
#undef FLT_2_0
#undef FLT_2_1
#undef FLT_2_2
#undef FLT_2_3
#undef FLT_2_4
#undef FLT_3_0
#undef FLT_3_1
#undef FLT_3_2
#undef FLT_3_3
#undef FLT_3_4
#undef FLT_HOLE
#undef FLT_4_0
#undef FLT_4_1
#undef FLT_4_2
#undef FLT_4_3
#undef FLT_4_4

void ConvPerChannelD1(
    const tflite::ConvParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  const int32_t input_offset = params.input_offset;  // r = s(q - Z)
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = tflite::MatchingDim(input_shape, 0, output_shape, 0);
  const int input_depth = input_shape.Dims(3);
  const int output_depth = tflite::MatchingDim(filter_shape, 0, output_shape, 3);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int filter_input_depth = filter_shape.Dims(3);
  const int groups = input_depth / filter_input_depth;
  TFLITE_DCHECK_NE(groups, 0);
  TFLITE_DCHECK_EQ(input_depth % filter_input_depth, 0);
  const int filters_per_group = output_depth / groups;
  TFLITE_DCHECK_NE(filters_per_group, 0);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  // Scratch pads to juggle data
  const size_t swizzled_filter_data_size = 32 * filter_height * filter_width;
  std::unique_ptr<int8_t> swizzled_filter_data(
      reinterpret_cast<int8_t*>(
          ::aligned_alloc(32, swizzled_filter_data_size)));
  int32_t swizzled_bias_data[32];
  int32_t swizzled_output_multiplier[32];
  int32_t swizzled_output_shift[32];

  for (int out_channel = 0; out_channel < output_depth; out_channel += 32) {
    int n_channels = std::min(32, output_depth - out_channel);

    // Transpose filter for easy loading
    for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
      for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
        for (int i = 0; i < n_channels; i++) {
          int filter_location =
              (filter_y * filter_width * 32) + (filter_x * 32) + i;
          swizzled_filter_data.get()[filter_location] = filter_data[
              tflite::Offset(filter_shape, out_channel + i, filter_y, filter_x,
                             0)];
        }
      }
    }

    if (bias_data) {
      JumptableSwizzle(bias_data + out_channel, swizzled_bias_data, n_channels);
      vld_w_x_m(v52, swizzled_bias_data);
    } else {
      vdup_w_x_m(v52, 0);
    }

    JumptableSwizzle(output_multiplier + out_channel,
                     swizzled_output_multiplier, n_channels);
    vld_w_x_m(v56, swizzled_output_multiplier);

    JumptableSwizzle(output_shift + out_channel, swizzled_output_shift,
                     n_channels);
    vld_w_x_m(v60, swizzled_output_shift);
    vrsub_w_vx_m(v60, v60, 0);

    int8_t* local_output_data = output_data + out_channel;

    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;

          // Accumulator loop
          vmv_v_m(v48, v52);
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + dilation_height_factor * filter_y;
            if ((in_y < 0) || (in_y >= input_height)) {
              continue;
            }

            const int8_t* local_input_data = input_data +
                tflite::Offset(input_shape, batch, in_y, 0, 0);
            int filter_x = 0;
            int in_x = in_x_origin;
            const int8_t* local_filter_data = swizzled_filter_data.get() +
                  (filter_y * filter_width * 32);
            while (in_x < 0) {
              filter_x++;
              in_x += dilation_width_factor;
              local_filter_data += 32;
            }
            for (; (filter_x < filter_width) && (in_x < input_width);
                 ++filter_x, in_x += dilation_width_factor,
                 local_filter_data += 32) {

              int16_t input_val = local_input_data[in_x];
              int16_t input_val16 = static_cast<int16_t>(
                  input_val + input_offset);
              vdup_h_x(v32, input_val16);

              vld_b_l_xx(v0, local_filter_data, n_channels);
              vaddw_h_vx(v0, v0, 0);

              // Multiply
              vmulw_w_vv(v4, v0, v32);
              vmulw_w_vv(v6, v1, v32);

              // Accumulate
              vadd_w_vv_m(v48, v48, v4);
            }
          }

          // Output pipeline
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60, output_activation_min, output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_l_xx(v48, local_output_data, n_channels);
          local_output_data += output_depth;
        }
      }
    }
  }
}

}  // namespace kelvin::opt