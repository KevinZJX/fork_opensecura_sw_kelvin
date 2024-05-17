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
    vdup_b_x(v27, -input_offset); \
    vdup_b_x(v28, -input_offset); \
    vdup_b_x(v29, -input_offset); \
    vdup_b_x(v30, -input_offset); \
    vdup_b_x(v31, -input_offset); \
  }
#define PAD_ROW_1(input_offset)   \
  {                               \
    vdup_b_x(v32, -input_offset); \
    vdup_b_x(v33, -input_offset); \
    vdup_b_x(v34, -input_offset); \
    vdup_b_x(v35, -input_offset); \
    vdup_b_x(v36, -input_offset); \
  }
#define PAD_ROW_2(input_offset)   \
  {                               \
    vdup_b_x(v37, -input_offset); \
    vdup_b_x(v38, -input_offset); \
    vdup_b_x(v39, -input_offset); \
    vdup_b_x(v40, -input_offset); \
    vdup_b_x(v41, -input_offset); \
  }
#define PAD_ROW_3(input_offset)   \
  {                               \
    vdup_b_x(v42, -input_offset); \
    vdup_b_x(v43, -input_offset); \
    vdup_b_x(v44, -input_offset); \
    vdup_b_x(v45, -input_offset); \
    vdup_b_x(v46, -input_offset); \
  }
#define PAD_ROW_4(input_offset)   \
  {                               \
    vdup_b_x(v47, -input_offset); \
    vdup_b_x(v48, -input_offset); \
    vdup_b_x(v49, -input_offset); \
    vdup_b_x(v50, -input_offset); \
    vdup_b_x(v51, -input_offset); \
  }

#define LOAD_ROW_0(p_input, input_width, in_y, in_x)         \
  {                                                          \
    const int8_t* p_row = p_input + (in_y[0] * input_width); \
    vdup_b_x(v27, *(p_row + in_x[0]));                       \
    vdup_b_x(v28, *(p_row + in_x[1]));                       \
    vdup_b_x(v29, *(p_row + in_x[2]));                       \
    vdup_b_x(v30, *(p_row + in_x[3]));                       \
    vdup_b_x(v31, *(p_row + in_x[4]));                       \
  }

#define LOAD_ROW_1(p_input, input_width, in_y, in_x)         \
  {                                                          \
    const int8_t* p_row = p_input + (in_y[1] * input_width); \
    vdup_b_x(v32, *(p_row + in_x[0]));                       \
    vdup_b_x(v33, *(p_row + in_x[1]));                       \
    vdup_b_x(v34, *(p_row + in_x[2]));                       \
    vdup_b_x(v35, *(p_row + in_x[3]));                       \
    vdup_b_x(v36, *(p_row + in_x[4]));                       \
  }

#define LOAD_ROW_2(p_input, input_width, in_y, in_x)         \
  {                                                          \
    const int8_t* p_row = p_input + (in_y[2] * input_width); \
    vdup_b_x(v37, *(p_row + in_x[0]));                       \
    vdup_b_x(v38, *(p_row + in_x[1]));                       \
    vdup_b_x(v39, *(p_row + in_x[2]));                       \
    vdup_b_x(v40, *(p_row + in_x[3]));                       \
    vdup_b_x(v41, *(p_row + in_x[4]));                       \
  }

#define LOAD_ROW_3(p_input, input_width, in_y, in_x)         \
  {                                                          \
    const int8_t* p_row = p_input + (in_y[3] * input_width); \
    vdup_b_x(v42, *(p_row + in_x[0]));                       \
    vdup_b_x(v43, *(p_row + in_x[1]));                       \
    vdup_b_x(v44, *(p_row + in_x[2]));                       \
    vdup_b_x(v45, *(p_row + in_x[3]));                       \
    vdup_b_x(v46, *(p_row + in_x[4]));                       \
  }

#define LOAD_ROW_4(p_input, input_width, in_y, in_x)         \
  {                                                          \
    const int8_t* p_row = p_input + (in_y[4] * input_width); \
    vdup_b_x(v47, *(p_row + in_x[0]));                       \
    vdup_b_x(v48, *(p_row + in_x[1]));                       \
    vdup_b_x(v49, *(p_row + in_x[2]));                       \
    vdup_b_x(v50, *(p_row + in_x[3]));                       \
    vdup_b_x(v51, *(p_row + in_x[4]));                       \
  }

#define H_PAD_OR_LOAD_ROW_0(p_input, input_width, input_offset, in_y, in_x) \
  if (in_x[0] >= 0 && in_x[4] < input_width) {                              \
    LOAD_ROW_0(p_input, input_width, in_y, in_x);                           \
  } else {                                                                  \
    const int8_t* p_row = p_input + (in_y[0] * input_width);                \
    if (in_x[0] < 0 || in_x[0] >= input_width) {                            \
      vdup_b_x(v27, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v27, *(p_row + in_x[0]));                                    \
    }                                                                       \
    if (in_x[1] < 0 || in_x[1] >= input_width) {                            \
      vdup_b_x(v28, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v28, *(p_row + in_x[1]));                                    \
    }                                                                       \
    if (in_x[2] < 0 || in_x[2] >= input_width) {                            \
      vdup_b_x(v29, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v29, *(p_row + in_x[2]));                                    \
    }                                                                       \
    if (in_x[3] < 0 || in_x[3] >= input_width) {                            \
      vdup_b_x(v30, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v30, *(p_row + in_x[3]));                                    \
    }                                                                       \
    if (in_x[4] < 0 || in_x[4] >= input_width) {                            \
      vdup_b_x(v31, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v31, *(p_row + in_x[4]));                                    \
    }                                                                       \
  }

#define H_PAD_OR_LOAD_ROW_1(p_input, input_width, input_offset, in_y, in_x) \
  if (in_x[0] >= 0 && in_x[4] < input_width) {                              \
    LOAD_ROW_1(p_input, input_width, in_y, in_x);                           \
  } else {                                                                  \
    const int8_t* p_row = p_input + (in_y[1] * input_width);                \
    if (in_x[0] < 0 || in_x[0] >= input_width) {                            \
      vdup_b_x(v32, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v32, *(p_row + in_x[0]));                                    \
    }                                                                       \
    if (in_x[1] < 0 || in_x[1] >= input_width) {                            \
      vdup_b_x(v33, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v33, *(p_row + in_x[1]));                                    \
    }                                                                       \
    if (in_x[2] < 0 || in_x[2] >= input_width) {                            \
      vdup_b_x(v34, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v34, *(p_row + in_x[2]));                                    \
    }                                                                       \
    if (in_x[3] < 0 || in_x[3] >= input_width) {                            \
      vdup_b_x(v35, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v35, *(p_row + in_x[3]));                                    \
    }                                                                       \
    if (in_x[4] < 0 || in_x[4] >= input_width) {                            \
      vdup_b_x(v36, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v36, *(p_row + in_x[4]));                                    \
    }                                                                       \
  }

#define H_PAD_OR_LOAD_ROW_2(p_input, input_width, input_offset, in_y, in_x) \
  if (in_x[0] >= 0 && in_x[4] < input_width) {                              \
    LOAD_ROW_2(p_input, input_width, in_y, in_x);                           \
  } else {                                                                  \
    const int8_t* p_row = p_input + (in_y[2] * input_width);                \
    if (in_x[0] < 0 || in_x[0] >= input_width) {                            \
      vdup_b_x(v37, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v37, *(p_row + in_x[0]));                                    \
    }                                                                       \
    if (in_x[1] < 0 || in_x[1] >= input_width) {                            \
      vdup_b_x(v38, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v38, *(p_row + in_x[1]));                                    \
    }                                                                       \
    if (in_x[2] < 0 || in_x[2] >= input_width) {                            \
      vdup_b_x(v39, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v39, *(p_row + in_x[2]));                                    \
    }                                                                       \
    if (in_x[3] < 0 || in_x[3] >= input_width) {                            \
      vdup_b_x(v40, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v40, *(p_row + in_x[3]));                                    \
    }                                                                       \
    if (in_x[4] < 0 || in_x[4] >= input_width) {                            \
      vdup_b_x(v41, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v41, *(p_row + in_x[4]));                                    \
    }                                                                       \
  }

#define H_PAD_OR_LOAD_ROW_3(p_input, input_width, input_offset, in_y, in_x) \
  if (in_x[0] >= 0 && in_x[4] < input_width) {                              \
    LOAD_ROW_3(p_input, input_width, in_y, in_x);                           \
  } else {                                                                  \
    const int8_t* p_row = p_input + (in_y[3] * input_width);                \
    if (in_x[0] < 0 || in_x[0] >= input_width) {                            \
      vdup_b_x(v42, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v42, *(p_row + in_x[0]));                                    \
    }                                                                       \
    if (in_x[1] < 0 || in_x[1] >= input_width) {                            \
      vdup_b_x(v43, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v43, *(p_row + in_x[1]));                                    \
    }                                                                       \
    if (in_x[2] < 0 || in_x[2] >= input_width) {                            \
      vdup_b_x(v44, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v44, *(p_row + in_x[2]));                                    \
    }                                                                       \
    if (in_x[3] < 0 || in_x[3] >= input_width) {                            \
      vdup_b_x(v45, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v45, *(p_row + in_x[3]));                                    \
    }                                                                       \
    if (in_x[4] < 0 || in_x[4] >= input_width) {                            \
      vdup_b_x(v46, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v46, *(p_row + in_x[4]));                                    \
    }                                                                       \
  }

#define H_PAD_OR_LOAD_ROW_4(p_input, input_width, input_offset, in_y, in_x) \
  if (in_x[0] >= 0 && in_x[4] < input_width) {                              \
    LOAD_ROW_4(p_input, input_width, in_y, in_x);                           \
  } else {                                                                  \
    const int8_t* p_row = p_input + (in_y[4] * input_width);                \
    if (in_x[0] < 0 || in_x[0] >= input_width) {                            \
      vdup_b_x(v47, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v47, *(p_row + in_x[0]));                                    \
    }                                                                       \
    if (in_x[1] < 0 || in_x[1] >= input_width) {                            \
      vdup_b_x(v48, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v48, *(p_row + in_x[1]));                                    \
    }                                                                       \
    if (in_x[2] < 0 || in_x[2] >= input_width) {                            \
      vdup_b_x(v49, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v49, *(p_row + in_x[2]));                                    \
    }                                                                       \
    if (in_x[3] < 0 || in_x[3] >= input_width) {                            \
      vdup_b_x(v50, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v50, *(p_row + in_x[3]));                                    \
    }                                                                       \
    if (in_x[4] < 0 || in_x[4] >= input_width) {                            \
      vdup_b_x(v51, -input_offset);                                         \
    } else {                                                                \
      vdup_b_x(v51, *(p_row + in_x[4]));                                    \
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
    adwconv_vxv(v60, v27, cmds, v0);      \
    adwconv_vxv(v60, v30, cmds, v3);      \
    adwconv_vxv(v60, v33, cmds, v6);      \
    adwconv_vxv(v60, v36, cmds, v9);      \
    adwconv_vxv(v60, v39, cmds, v12);     \
    adwconv_vxv(v60, v42, cmds, v15);     \
    adwconv_vxv(v60, v45, cmds, v18);     \
    adwconv_vxv(v60, v48, cmds, v21);     \
    vdwconv_vxv(v60, v51, cmds, v24);     \
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
  vld_b_l_xx(v0, p_flt_0 + (0 * 24), 24);
  vld_b_l_xx(v1, p_flt_0 + (1 * 24), 24);
  vld_b_l_xx(v2, p_flt_0 + (2 * 24), 24);
  vld_b_l_xx(v3, p_flt_0 + (3 * 24), 24);
  vld_b_l_xx(v4, p_flt_0 + (4 * 24), 24);

  vld_b_l_xx(v5, p_flt_1 + (0 * 24), 24);
  vld_b_l_xx(v6, p_flt_1 + (1 * 24), 24);
  vld_b_l_xx(v7, p_flt_1 + (2 * 24), 24);
  vld_b_l_xx(v8, p_flt_1 + (3 * 24), 24);
  vld_b_l_xx(v9, p_flt_1 + (4 * 24), 24);

  vld_b_l_xx(v10, p_flt_2 + (0 * 24), 24);
  vld_b_l_xx(v11, p_flt_2 + (1 * 24), 24);
  vld_b_l_xx(v12, p_flt_2 + (2 * 24), 24);
  vld_b_l_xx(v13, p_flt_2 + (3 * 24), 24);
  vld_b_l_xx(v14, p_flt_2 + (4 * 24), 24);

  vld_b_l_xx(v15, p_flt_3 + (0 * 24), 24);
  vld_b_l_xx(v16, p_flt_3 + (1 * 24), 24);
  vld_b_l_xx(v17, p_flt_3 + (2 * 24), 24);
  vld_b_l_xx(v18, p_flt_3 + (3 * 24), 24);
  vld_b_l_xx(v19, p_flt_3 + (4 * 24), 24);

  vld_b_l_xx(v20, p_flt_4 + (0 * 24), 24);
  vld_b_l_xx(v21, p_flt_4 + (1 * 24), 24);
  vld_b_l_xx(v22, p_flt_4 + (2 * 24), 24);
  vld_b_l_xx(v23, p_flt_4 + (3 * 24), 24);
  vld_b_l_xx(v24, p_flt_4 + (4 * 24), 24);
  vdup_b_x(v25, 0);
  vdup_b_x(v26, 0);

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
  int in_x[5];
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