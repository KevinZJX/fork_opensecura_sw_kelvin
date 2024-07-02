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

// Depthwise convolution based on Kelvin ops
// Data types: input: s8, filter: s8, bias s32

#include "tensorflow/lite/kernels/internal/reference/integer_ops/depthwise_conv.h"
#include "tflm/opt/conv_util.h"

namespace kelvin::opt {
namespace {

// Reorders a vector to match the pattern after double-widening.
// N must be a multiple of 4.
// Working only for mutliples of 32
void VectorSwizzle(const int32_t* input, int32_t* output, int N) {
  assert(N >= 4 && N % 4 == 0);
  const int32_t(&in)[N] = *(int32_t(*)[N])input;
  int32_t(&out)[N] = *(int32_t(*)[N]) output;
  const int32_t* p_in = in;
  for (int i = 0; i < N / 4; ++i) {
    int32_t* out0 = out + i + 0;
    int32_t* out1 = out + i + 16;
    int32_t* out2 = out + i + 8;
    int32_t* out3 = out + i + 24;
    *out0 = *p_in++;
    *out1 = *p_in++;
    *out2 = *p_in++;
    *out3 = *p_in++;
  }
}

// special case of input depth = 32n, filter shape of 3x3, strides of 1
void DepthwiseConvS83x3D32_Stride1(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data
) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);
  int32_t swizzled_bias_data[32];
  int32_t swizzled_shift_multi[32];
  int32_t swizzled_output_multi[32];

  for (int in_channel = 0; in_channel + 32 <= input_depth; in_channel += 32) {
    const int output_channel = in_channel;
    int8_t* p_output = output_data + output_channel;
    VectorSwizzle(bias_data + output_channel, swizzled_bias_data, 32);
    VectorSwizzle(output_multiplier + output_channel, swizzled_output_multi, 32);
    VectorSwizzle(output_shift + output_channel, swizzled_shift_multi, 32);

    vld_w_x_m(v52, swizzled_bias_data);
    vld_w_x_m(v56, swizzled_output_multi);
    vld_w_x_m(v60, swizzled_shift_multi);
    vrsub_w_vx_m(v60, v60, 0);

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

#define FLT_0_0 v0
#define FLT_0_1 v3
#define FLT_0_2 v6
#define FLT_1_0 v1
#define FLT_1_1 v4
#define FLT_1_2 v7
#define FLT_2_0 v2
#define FLT_2_1 v5
#define FLT_2_2 v8

#define INPUT_0_0 v9
#define INPUT_0_1 v12
#define INPUT_0_2 v15
#define INPUT_0_3 v18
#define INPUT_0_4 v21
#define INPUT_0_5 v24
#define INPUT_1_0 v10
#define INPUT_1_1 v13
#define INPUT_1_2 v16
#define INPUT_1_3 v19
#define INPUT_1_4 v22
#define INPUT_1_5 v25
#define INPUT_2_0 v11
#define INPUT_2_1 v14
#define INPUT_2_2 v17
#define INPUT_2_3 v20
#define INPUT_2_4 v23
#define INPUT_2_5 v26

#define INPUT_PTRS(_strides) \
  const int in_y_origin = (out_y * stride_height) - pad_height; \
  const int in_x_origin = (out_x * stride_width) - pad_width; \
  const int8_t* p_in_0 = input_data + \
      (batch * input_height * input_width * input_depth) + \
      (in_y_origin * input_width * input_depth) + \
      ((in_x_origin + _strides) * input_depth) + \
      in_channel; \
  const int8_t* p_in_1 = p_in_0 + (input_width * input_depth); \
  const int8_t* p_in_2 = p_in_1 + (input_width * input_depth); \
  (void)p_in_2;

#define COMPUTE() \
  adwinit_v(v48, v48); \
  adwconv_vxv(v48, INPUT_0_0, cmds, FLT_0_0); \
  adwconv_vxv(v48, INPUT_0_1, cmds, FLT_0_1); \
  vdwconv_vxv(v48, INPUT_0_2, cmds, FLT_0_2);

    // Don't reorder me, otherwise data will not be
    // loaded in the correct order
    // (we can reuse the p_flt* due to the `p` vld variant).
    const int8_t* p_flt0 = filter_data + in_channel;
    const int8_t* p_flt1 = p_flt0 + input_depth;
    const int32_t stride = 2 * input_depth;
    vld_b_sp_xx(FLT_0_0, p_flt0, stride);
    vld_b_sp_xx(FLT_0_1, p_flt1, stride);
    vld_b_sp_xx(FLT_0_2, p_flt0, stride);
    vld_b_sp_xx(FLT_1_0, p_flt1, stride);
    vld_b_sp_xx(FLT_1_1, p_flt0, stride);
    vld_b_sp_xx(FLT_1_2, p_flt1, stride);
    vld_b_sp_xx(FLT_2_0, p_flt0, stride);
    vld_b_sp_xx(FLT_2_1, p_flt1, stride);
    vld_b_sp_xx(FLT_2_2, p_flt0, stride);

    for (int batch = 0; batch < batches; ++batch) {
      int out_y = 0;
      for (; out_y < pad_height; ++out_y) {
        int out_x = 0;
        vdup_b_x(INPUT_0_0, -input_offset);
        vdup_b_x(INPUT_0_1, -input_offset);
        vdup_b_x(INPUT_0_2, -input_offset);
        for (; out_x < pad_width; ++out_x) {
          INPUT_PTRS(1);
          vmv_v_m(v48, v52);

          vdup_b_x(INPUT_1_0, -input_offset);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vdup_b_x(INPUT_2_0, -input_offset);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);

          p_output += output_depth;
        }
        for (; out_x < output_width - pad_width; ++out_x) {
          INPUT_PTRS(0);
          vmv_v_m(v48, v52);
          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);
          p_output += output_depth;
        }
        for (; out_x < output_width; ++out_x) {
          INPUT_PTRS(0);
          vmv_v_m(v48, v52);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vdup_b_x(INPUT_1_2, -input_offset);
          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vdup_b_x(INPUT_2_2, -input_offset);

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);

          p_output += output_depth;
        }
      }
      for (; out_y < output_height - pad_height; ++out_y) {
        int out_x = 0;
        for (; out_x < pad_width; ++out_x) {
          INPUT_PTRS(1);
          vmv_v_m(v48, v52);

          vdup_b_x(INPUT_0_0, -input_offset);
          vdup_b_x(INPUT_1_0, -input_offset);
          vdup_b_x(INPUT_2_0, -input_offset);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);

          p_output += output_depth;
        }
        for (; out_x + 4 <= output_width - pad_width; out_x += 4) {
          INPUT_PTRS(0);
          // Initialize accumulators w/ bias data.
          vmv_v_m(v36, v52);
          vmv_v_m(v40, v52);
          vmv_v_m(v44, v52);
          vmv_v_m(v48, v52);

          vld_b_sp_xx(INPUT_0_0, p_in_0, stride_width * input_depth);
          vld_b_sp_xx(INPUT_1_0, p_in_1, stride_width * input_depth);
          vld_b_sp_xx(INPUT_2_0, p_in_2, stride_width * input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, stride_width * input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, stride_width * input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, stride_width * input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, stride_width * input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, stride_width * input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, stride_width * input_depth);

          adwinit_v(v48, v48);
          adwconv_vxv(v48, INPUT_0_0, cmds, FLT_0_0);
          adwconv_vxv(v48, INPUT_0_1, cmds, FLT_0_1);
          vdwconv_vxv(v48, INPUT_0_2, cmds, FLT_0_2);

          vld_b_sp_xx(INPUT_0_3, p_in_0, stride_width * input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, stride_width * input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, stride_width * input_depth);

          adwinit_v(v44, v44);
          adwconv_vxv(v44, INPUT_0_1, cmds, FLT_0_0);
          adwconv_vxv(v44, INPUT_0_2, cmds, FLT_0_1);
          vdwconv_vxv(v44, INPUT_0_3, cmds, FLT_0_2);

          vld_b_sp_xx(INPUT_0_4, p_in_0, stride_width * input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, stride_width * input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, stride_width * input_depth);

          adwinit_v(v40, v40);
          adwconv_vxv(v40, INPUT_0_2, cmds, FLT_0_0);
          adwconv_vxv(v40, INPUT_0_3, cmds, FLT_0_1);
          vdwconv_vxv(v40, INPUT_0_4, cmds, FLT_0_2);

          vld_b_sp_xx(INPUT_0_5, p_in_0, stride_width * input_depth);
          vld_b_sp_xx(INPUT_1_5, p_in_1, stride_width * input_depth);
          vld_b_sp_xx(INPUT_2_5, p_in_2, stride_width * input_depth);

          adwinit_v(v36, v36);
          adwconv_vxv(v36, INPUT_0_3, cmds, FLT_0_0);
          adwconv_vxv(v36, INPUT_0_4, cmds, FLT_0_1);
          vdwconv_vxv(v36, INPUT_0_5, cmds, FLT_0_2);

          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE4(
              v48, v44, v40, v36, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vsraqs_b_vx(v44, v44, 0);
          vsraqs_b_vx(v40, v40, 0);
          vsraqs_b_vx(v36, v36, 0);

          vst_b_x(v48, p_output);
          p_output += output_depth;

          vst_b_x(v44, p_output);
          p_output += output_depth;

          vst_b_x(v40, p_output);
          p_output += output_depth;

          vst_b_x(v36, p_output);
          p_output += output_depth;
        }
        for (; out_x < output_width - pad_width; ++out_x) {
          INPUT_PTRS(0);
          vmv_v_m(v48, v52);

          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);
          p_output += output_depth;
        }
        for (; out_x < output_width; ++out_x) {
          INPUT_PTRS(0);
          vmv_v_m(v48, v52);

          vdup_b_x(INPUT_0_2, -input_offset);
          vdup_b_x(INPUT_1_2, -input_offset);
          vdup_b_x(INPUT_2_2, -input_offset);
          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);

          p_output += output_depth;
        }
      }
      for (; out_y < output_height; ++out_y) {
        vdup_b_x(INPUT_2_0, -input_offset);
        vdup_b_x(INPUT_2_1, -input_offset);
        vdup_b_x(INPUT_2_2, -input_offset);
        int out_x = 0;
        for (; out_x < pad_width; ++out_x) {
          INPUT_PTRS(1);
          vmv_v_m(v48, v52);

          vdup_b_x(INPUT_0_0, -input_offset);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vdup_b_x(INPUT_1_0, -input_offset);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);
          p_output += output_depth;
        }
        for (; out_x < output_width - pad_width; ++out_x) {
          INPUT_PTRS(0);
          vmv_v_m(v48, v52);
          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);
          p_output += output_depth;
        }
        for (; out_x < output_width; ++out_x) {
          INPUT_PTRS(0);
          vmv_v_m(v48, v52);

          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vdup_b_x(INPUT_0_2, -input_offset);
          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vdup_b_x(INPUT_1_2, -input_offset);

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);
          p_output += output_depth;
        }
      }
    }
  }
#undef FLT_0_0
#undef FLT_0_1
#undef FLT_0_2
#undef FLT_1_0
#undef FLT_1_1
#undef FLT_1_2
#undef FLT_2_0
#undef FLT_2_1
#undef FLT_2_2
#undef INPUT_0_0
#undef INPUT_0_1
#undef INPUT_0_2
#undef INPUT_0_3
#undef INPUT_0_4
#undef INPUT_0_5
#undef INPUT_1_0
#undef INPUT_1_1
#undef INPUT_1_2
#undef INPUT_1_3
#undef INPUT_1_4
#undef INPUT_1_5
#undef INPUT_2_0
#undef INPUT_2_1
#undef INPUT_2_2
#undef INPUT_2_3
#undef INPUT_2_4
#undef INPUT_2_5
#undef COMPUTE
#undef INPUT_PTRS
}

// special case of input depth = 32n, filter shape of 3x3
void DepthwiseConvS83x3D32(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data
) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);
  int32_t swizzled_bias_data[32];
  int32_t swizzled_shift_multi[32];
  int32_t swizzled_output_multi[32];

  for (int in_channel = 0; in_channel + 32 <= input_depth; in_channel += 32) {
    const int output_channel = in_channel;
    VectorSwizzle(bias_data + output_channel, swizzled_bias_data, 32);
    VectorSwizzle(output_multiplier + output_channel, swizzled_output_multi, 32);
    VectorSwizzle(output_shift + output_channel, swizzled_shift_multi, 32);

    vld_w_x_m(v52, swizzled_bias_data);
    vld_w_x_m(v56, swizzled_output_multi);
    vld_w_x_m(v60, swizzled_shift_multi);
    vrsub_w_vx_m(v60, v60, 0);

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

    // Don't reorder me, otherwise data will not be
    // loaded in the correct order
    // (we can reuse the p_flt* due to the `p` vld variant).
    const int8_t* p_flt0 = filter_data + in_channel;
    const int8_t* p_flt1 = p_flt0 + input_depth;
    const int32_t stride = 2 * input_depth;
    vld_b_sp_xx(v6, p_flt0, stride);
    vld_b_sp_xx(v7, p_flt1, stride);
    vld_b_sp_xx(v8, p_flt0, stride);
    vld_b_sp_xx(v9, p_flt1, stride);
    vld_b_sp_xx(v10, p_flt0, stride);
    vld_b_sp_xx(v11, p_flt1, stride);
    vld_b_sp_xx(v12, p_flt0, stride);
    vld_b_sp_xx(v13, p_flt1, stride);
    vld_b_sp_xx(v14, p_flt0, stride);

    for (int batch = 0; batch < batches; ++batch) {
      const int8_t* p_output = output_data + (batch * output_width * output_height * output_depth) + output_channel;
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        const int y_offset = (output_depth * output_width * out_y);
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;

          // Initialize accumulators w/ bias data.
          vmv_v_m(v48, v52);

          bool top_pad = in_y_origin < 0;
          bool left_pad = in_x_origin < 0;
          bool bottom_pad = (in_y_origin + 2) >= input_height;
          bool right_pad = (in_x_origin + 2) >= input_width;
          bool padding_required = top_pad || left_pad || bottom_pad || right_pad;
          const int8_t* p_in_0 = input_data +
            (batch * input_height * input_width * input_depth) +
            (in_y_origin * input_width * input_depth) +
            (in_x_origin * input_depth) +
            in_channel;
          const int8_t* p_in_1 = p_in_0 + (input_width * input_depth);
          const int8_t* p_in_2 = p_in_1 + (input_width * input_depth);
          if (!padding_required) {
            vld_b_sp_xx(v15, p_in_0, input_depth);
            vld_b_sp_xx(v16, p_in_0, input_depth);
            vld_b_sp_xx(v17, p_in_0, input_depth);
            vld_b_sp_xx(v18, p_in_1, input_depth);
            vld_b_sp_xx(v19, p_in_1, input_depth);
            vld_b_sp_xx(v20, p_in_1, input_depth);
            vld_b_sp_xx(v21, p_in_2, input_depth);
            vld_b_sp_xx(v22, p_in_2, input_depth);
            vld_b_sp_xx(v23, p_in_2, input_depth);
          } else {
            // Top row
            if (top_pad || left_pad) {
              vdup_b_x(v15, -input_offset);
            } else {
              vld_b_x(v15, p_in_0);
            }
            if (top_pad) {
              vdup_b_x(v16, -input_offset);
            } else {
              vld_b_x(v16, p_in_0 + input_depth);
            }
            if (top_pad || right_pad) {
              vdup_b_x(v17, -input_offset);
            } else {
              vld_b_x(v17, p_in_0 + (2 * input_depth));
            }
            // Middle row
            if (left_pad) {
              vdup_b_x(v18, -input_offset);
            } else {
              vld_b_x(v18, p_in_1);
            }
            vld_b_x(v19, p_in_1 + input_depth);
            if (right_pad) {
              vdup_b_x(v20, -input_offset);
            } else {
              vld_b_x(v20, p_in_1 + (2 * input_depth));
            }
            // Bottom row
            if (bottom_pad || left_pad) {
              vdup_b_x(v21, -input_offset);
            } else {
              vld_b_x(v21, p_in_2);
            }
            if (bottom_pad) {
              vdup_b_x(v22, -input_offset);
            } else {
              vld_b_x(v22, p_in_2 + input_depth);
            }
            if (bottom_pad || right_pad) {
              vdup_b_x(v23, -input_offset);
            } else {
              vld_b_x(v23, p_in_2 + (2 * input_depth));
            }
          }

          adwinit_v(v48, v48);
          adwconv_vxv(v48, v15, cmds, v6);
          adwconv_vxv(v48, v18, cmds, v9);
          vdwconv_vxv(v48, v21, cmds, v12);

          vdmulh_w_rn_vv_m(v48, v48, v56);
          vsha_w_r_vv_m(v48, v48, v60);
          vadd_w_vx_m(v48, v48, output_offset);
          vmax_w_vx_m(v48, v48, output_activation_min);
          vmin_w_vx_m(v48, v48, output_activation_max);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output + (out_x * output_depth) + y_offset);
        }
      }
    }
  }
}

// special case of input depth = 32n, filter shape of 5x5, stride == 1
void DepthwiseConvS85x5D32_Stride1(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data
) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  assert(pad_width == 2);
  assert(pad_height == 2);
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);
  int32_t swizzled_bias_data[32];
  int32_t swizzled_shift_multi[32];
  int32_t swizzled_output_multi[32];

// INPUT_Y_X
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
#define INPUT_4_0 v47
#define INPUT_4_1 v48
#define INPUT_4_2 v49
#define INPUT_4_3 v50
#define INPUT_4_4 v51

#define INPUT_0_5 v53
#define INPUT_1_5 v54
#define INPUT_2_5 v55
#define INPUT_3_5 v46
#define INPUT_4_5 v52

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

#define COMPUTE() \
  vld_w_x_m(v60, swizzled_bias_data); \
  adwinit_v(v60, v60); \
  /* 0,0 1,0 2,0 */ \
  adwconv_vxv(v60, INPUT_0_0, cmds, FLT_0_0); \
  /* 0,1 1,1 2,1 */ \
  adwconv_vxv(v60, INPUT_0_1, cmds, FLT_0_1); \
  /* 0,2 1,2 2,2*/ \
  adwconv_vxv(v60, INPUT_0_2, cmds, FLT_0_2); \
  /* 0,3 1,3 2,3 */ \
  adwconv_vxv(v60, INPUT_0_3, cmds, FLT_0_3); \
  /* 0,4 1,4 2,4 */ \
  adwconv_vxv(v60, INPUT_0_4, cmds, FLT_0_4); \
  /* 3,0 3,1 3,2 */ \
  adwconv_vxv(v60, INPUT_3_0, cmds, FLT_3_0); \
  /* 3,3 3,4 hole */ \
  adwconv_vxv(v60, INPUT_3_3, cmds, FLT_3_3); \
  /* hole 4,0 4,1*/ \
  adwconv_vxv(v60, INPUT_3_5, cmds, FLT_HOLE); \
  /* 4,2 4,3 4,4*/ \
  vdwconv_vxv(v60, INPUT_4_2, cmds, FLT_4_2); \
  INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(v60, v56, v52, \
      output_activation_min, \
      output_activation_max, \
      output_offset); \
  vsraqs_b_vx(v60, v60, 0); \
  vst_b_x(v60, p_output);

#define INPUT_PTRS(_strides) \
  const int in_x_origin = (out_x * stride_width) - pad_width; \
  const int in_y_origin = (out_y * stride_height) - pad_height; \
  const int8_t* p_in_0 = input_data + \
    (batch * input_height * input_width * input_depth) + \
    (in_y_origin * input_width * input_depth) + \
    ((in_x_origin + _strides) * input_depth) + \
    in_channel; \
  const int8_t* p_in_1 = p_in_0 + (input_width * input_depth); \
  const int8_t* p_in_2 = p_in_1 + (input_width * input_depth); \
  const int8_t* p_in_3 = p_in_2 + (input_width * input_depth); \
  const int8_t* p_in_4 = p_in_3 + (input_width * input_depth); \
  (void)p_in_4;

  for (int in_channel = 0; in_channel + 32 <= input_depth; in_channel += 32) {
    const int output_channel = in_channel;
    VectorSwizzle(bias_data + output_channel, swizzled_bias_data, 32);
    VectorSwizzle(output_multiplier + output_channel, swizzled_output_multi, 32);
    VectorSwizzle(output_shift + output_channel, swizzled_shift_multi, 32);

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

    // Don't reorder me!
    const int8_t* p_flt0 = filter_data + in_channel;
    const int32_t stride = input_depth;
    vld_b_sp_xx(FLT_0_0, p_flt0, stride);
    vld_b_sp_xx(FLT_0_1, p_flt0, stride);
    vld_b_sp_xx(FLT_0_2, p_flt0, stride);
    vld_b_sp_xx(FLT_0_3, p_flt0, stride);
    vld_b_sp_xx(FLT_0_4, p_flt0, stride);
    vld_b_sp_xx(FLT_1_0, p_flt0, stride);
    vld_b_sp_xx(FLT_1_1, p_flt0, stride);
    vld_b_sp_xx(FLT_1_2, p_flt0, stride);
    vld_b_sp_xx(FLT_1_3, p_flt0, stride);
    vld_b_sp_xx(FLT_1_4, p_flt0, stride);
    vld_b_sp_xx(FLT_2_0, p_flt0, stride);
    vld_b_sp_xx(FLT_2_1, p_flt0, stride);
    vld_b_sp_xx(FLT_2_2, p_flt0, stride);
    vld_b_sp_xx(FLT_2_3, p_flt0, stride);
    vld_b_sp_xx(FLT_2_4, p_flt0, stride);
    vld_b_sp_xx(FLT_3_0, p_flt0, stride);
    vld_b_sp_xx(FLT_3_1, p_flt0, stride);
    vld_b_sp_xx(FLT_3_2, p_flt0, stride);
    vld_b_sp_xx(FLT_3_3, p_flt0, stride);
    vld_b_sp_xx(FLT_3_4, p_flt0, stride);
    vld_b_sp_xx(FLT_4_0, p_flt0, stride);
    vld_b_sp_xx(FLT_4_1, p_flt0, stride);
    vld_b_sp_xx(FLT_4_2, p_flt0, stride);
    vld_b_sp_xx(FLT_4_3, p_flt0, stride);
    vld_b_sp_xx(FLT_4_4, p_flt0, stride);
    vdup_b_x(FLT_HOLE, 0);

    vld_w_x_m(v56, swizzled_output_multi);
    vld_w_x_m(v52, swizzled_shift_multi);
    vrsub_w_vx_m(v52, v52, 0);
    for (int batch = 0; batch < batches; ++batch) {
      const int8_t* p_output = output_data + (batch * output_height * output_width * output_depth) + output_channel;
      int out_y = 0;
      // Done
      { // out_y = 0;
        int out_x = 0;
        vdup_b_x(INPUT_0_0, -input_offset);
        vdup_b_x(INPUT_0_1, -input_offset);
        vdup_b_x(INPUT_0_2, -input_offset);
        vdup_b_x(INPUT_0_3, -input_offset);
        vdup_b_x(INPUT_0_4, -input_offset);

        vdup_b_x(INPUT_1_0, -input_offset);
        vdup_b_x(INPUT_1_1, -input_offset);
        vdup_b_x(INPUT_1_2, -input_offset);
        vdup_b_x(INPUT_1_3, -input_offset);
        vdup_b_x(INPUT_1_4, -input_offset);
        { // out_x == 0
          INPUT_PTRS(2);

          vdup_b_x(INPUT_2_0, -input_offset);
          vdup_b_x(INPUT_2_1, -input_offset);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          vdup_b_x(INPUT_3_0, -input_offset);
          vdup_b_x(INPUT_3_1, -input_offset);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_in_3, input_depth);

          vdup_b_x(INPUT_4_0, -input_offset);
          vdup_b_x(INPUT_4_1, -input_offset);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_4, p_in_4, input_depth);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        { // out_x == 1
          INPUT_PTRS(1);

          vdup_b_x(INPUT_2_0, -input_offset);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          vdup_b_x(INPUT_3_0, -input_offset);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_in_3, input_depth);

          vdup_b_x(INPUT_4_0, -input_offset);
          vld_b_sp_xx(INPUT_4_1, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_4, p_in_4, input_depth);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        for (; out_x < output_width - pad_width; ++out_x) {
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          vld_b_sp_xx(INPUT_3_0, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_in_3, input_depth);

          vld_b_sp_xx(INPUT_4_0, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_1, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_4, p_in_4, input_depth);
          COMPUTE();
          p_output += output_depth;
        }
        { // out_x == output_width - 2
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vdup_b_x(INPUT_2_4, -input_offset);

          vld_b_sp_xx(INPUT_3_0, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vdup_b_x(INPUT_3_4, -input_offset);

          vld_b_sp_xx(INPUT_4_0, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_1, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_in_4, input_depth);
          vdup_b_x(INPUT_4_4, -input_offset);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        { // out_x == output_width - 1
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vdup_b_x(INPUT_2_3, -input_offset);
          vdup_b_x(INPUT_2_4, -input_offset);

          vld_b_sp_xx(INPUT_3_0, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vdup_b_x(INPUT_3_3, -input_offset);
          vdup_b_x(INPUT_3_4, -input_offset);

          vld_b_sp_xx(INPUT_4_0, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_1, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vdup_b_x(INPUT_4_3, -input_offset);
          vdup_b_x(INPUT_4_4, -input_offset);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        ++out_y;
      }
      // Done
      { // out_y = 1;
        int out_x = 0;
        vdup_b_x(INPUT_0_0, -input_offset);
        vdup_b_x(INPUT_0_1, -input_offset);
        vdup_b_x(INPUT_0_2, -input_offset);
        vdup_b_x(INPUT_0_3, -input_offset);
        vdup_b_x(INPUT_0_4, -input_offset);
        {  // out_x = 0;
          INPUT_PTRS(2);

          vdup_b_x(INPUT_1_0, -input_offset);
          vdup_b_x(INPUT_1_1, -input_offset);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, input_depth);

          vdup_b_x(INPUT_2_0, -input_offset);
          vdup_b_x(INPUT_2_1, -input_offset);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          vdup_b_x(INPUT_3_0, -input_offset);
          vdup_b_x(INPUT_3_1, -input_offset);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_in_3, input_depth);

          vdup_b_x(INPUT_4_0, -input_offset);
          vdup_b_x(INPUT_4_1, -input_offset);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_4, p_in_4, input_depth);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        {  // out_x = 1;
          INPUT_PTRS(1);

          vdup_b_x(INPUT_1_0, -input_offset);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, input_depth);

          vdup_b_x(INPUT_2_0, -input_offset);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          vdup_b_x(INPUT_3_0, -input_offset);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_in_3, input_depth);

          vdup_b_x(INPUT_4_0, -input_offset);
          vld_b_sp_xx(INPUT_4_1, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_4, p_in_4, input_depth);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        for (; out_x < output_width - pad_width; ++out_x) {
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, input_depth);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          vld_b_sp_xx(INPUT_3_0, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_in_3, input_depth);

          vld_b_sp_xx(INPUT_4_0, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_1, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_4, p_in_4, input_depth);

          COMPUTE();
          p_output += output_depth;
        }
        { // out_x = output_width - 2
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vdup_b_x(INPUT_1_4, -input_offset);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vdup_b_x(INPUT_2_4, -input_offset);

          vld_b_sp_xx(INPUT_3_0, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vdup_b_x(INPUT_3_4, -input_offset);

          vld_b_sp_xx(INPUT_4_0, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_1, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_in_4, input_depth);
          vdup_b_x(INPUT_4_4, -input_offset);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        { // out_x = output_width - 1
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vdup_b_x(INPUT_1_3, -input_offset);
          vdup_b_x(INPUT_1_4, -input_offset);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vdup_b_x(INPUT_2_3, -input_offset);
          vdup_b_x(INPUT_2_4, -input_offset);

          vld_b_sp_xx(INPUT_3_0, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vdup_b_x(INPUT_3_3, -input_offset);
          vdup_b_x(INPUT_3_4, -input_offset);

          vld_b_sp_xx(INPUT_4_0, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_1, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vdup_b_x(INPUT_4_3, -input_offset);
          vdup_b_x(INPUT_4_4, -input_offset);

          COMPUTE();
          p_output += output_depth;
        }
        ++out_y;
      }
      // Done
      for (; out_y < output_height - pad_height; ++out_y) {
        int out_x = 0;
        { // out_x == 0
          INPUT_PTRS(2);

          vdup_b_x(INPUT_0_0, -input_offset);
          vdup_b_x(INPUT_0_1, -input_offset);
          vdup_b_x(INPUT_1_0, -input_offset);
          vdup_b_x(INPUT_1_1, -input_offset);
          vdup_b_x(INPUT_2_0, -input_offset);
          vdup_b_x(INPUT_2_1, -input_offset);
          vdup_b_x(INPUT_3_0, -input_offset);
          vdup_b_x(INPUT_3_1, -input_offset);
          vdup_b_x(INPUT_4_0, -input_offset);
          vdup_b_x(INPUT_4_1, -input_offset);

          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_4, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_4, p_in_4, input_depth);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        { // out_x == 1
          INPUT_PTRS(1);
          vdup_b_x(INPUT_0_0, -input_offset);
          vdup_b_x(INPUT_1_0, -input_offset);
          vdup_b_x(INPUT_2_0, -input_offset);
          vdup_b_x(INPUT_3_0, -input_offset);
          vdup_b_x(INPUT_4_0, -input_offset);

          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_4, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_4_1, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_4, p_in_4, input_depth);

          COMPUTE();
          p_output += input_depth;
          ++out_x;
        }
        for (; out_x + 4 <= output_width - pad_width; out_x += 4) {
          INPUT_PTRS(0);

          vld_w_x_m(v60, swizzled_bias_data);
          adwinit_v(v60, v60);
          // Load top 3x8, in column-major order
          vld_b_sp_xx(v26, p_in_0, input_depth);
          vld_b_sp_xx(v27, p_in_1, input_depth);
          vld_b_sp_xx(v28, p_in_2, input_depth);
          vld_b_sp_xx(v29, p_in_0, input_depth);
          vld_b_sp_xx(v30, p_in_1, input_depth);
          vld_b_sp_xx(v31, p_in_2, input_depth);
          vld_b_sp_xx(v32, p_in_0, input_depth);
          vld_b_sp_xx(v33, p_in_1, input_depth);
          vld_b_sp_xx(v34, p_in_2, input_depth);
          vld_b_sp_xx(v35, p_in_0, input_depth);
          vld_b_sp_xx(v36, p_in_1, input_depth);
          vld_b_sp_xx(v37, p_in_2, input_depth);
          vld_b_sp_xx(v38, p_in_0, input_depth);
          vld_b_sp_xx(v39, p_in_1, input_depth);
          vld_b_sp_xx(v40, p_in_2, input_depth);
          vld_b_sp_xx(v41, p_in_0, input_depth);
          vld_b_sp_xx(v42, p_in_1, input_depth);
          vld_b_sp_xx(v43, p_in_2, input_depth);
          vld_b_sp_xx(v44, p_in_0, input_depth);
          vld_b_sp_xx(v45, p_in_1, input_depth);
          vld_b_sp_xx(v46, p_in_2, input_depth);
          vld_b_sp_xx(v47, p_in_0, input_depth);
          vld_b_sp_xx(v48, p_in_1, input_depth);
          vld_b_sp_xx(v49, p_in_2, input_depth);

          // Compute 3x5, starting from 0,3
          adwconv_vxv(v60, v35, cmds, FLT_0_0);
          adwconv_vxv(v60, v38, cmds, FLT_0_1);
          adwconv_vxv(v60, v41, cmds, FLT_0_2);
          adwconv_vxv(v60, v44, cmds, FLT_0_3);
          vdwconv_vxv(v60, v47, cmds, FLT_0_4);

          // Compute 3x5, starting from 0,2
          vld_w_x_m(v56, swizzled_bias_data);
          adwinit_v(v56, v56);
          adwconv_vxv(v56, v32, cmds, FLT_0_0);
          adwconv_vxv(v56, v35, cmds, FLT_0_1);
          adwconv_vxv(v56, v38, cmds, FLT_0_2);
          adwconv_vxv(v56, v41, cmds, FLT_0_3);
          vdwconv_vxv(v56, v44, cmds, FLT_0_4);

          // Compute 3x5, starting from 0,1
          vld_w_x_m(v52, swizzled_bias_data);
          adwinit_v(v52, v52);
          adwconv_vxv(v52, v29, cmds, FLT_0_0);
          adwconv_vxv(v52, v32, cmds, FLT_0_1);
          adwconv_vxv(v52, v35, cmds, FLT_0_2);
          adwconv_vxv(v52, v38, cmds, FLT_0_3);
          vdwconv_vxv(v52, v41, cmds, FLT_0_4);

          // Compute 3x5, starting from 0,3
          vld_w_x_m(v48, swizzled_bias_data);
          adwinit_v(v48, v48);
          adwconv_vxv(v48, v26, cmds, FLT_0_0);
          adwconv_vxv(v48, v29, cmds, FLT_0_1);
          adwconv_vxv(v48, v32, cmds, FLT_0_2);
          adwconv_vxv(v48, v35, cmds, FLT_0_3);
          vdwconv_vxv(v48, v38, cmds, FLT_0_4);

          // Load bottom 2x8, row major
          vld_b_sp_xx(v26, p_in_3, input_depth);
          vld_b_sp_xx(v27, p_in_3, input_depth);
          vld_b_sp_xx(v28, p_in_3, input_depth);
          vld_b_sp_xx(v29, p_in_3, input_depth);
          vld_b_sp_xx(v30, p_in_3, input_depth);
          vld_b_sp_xx(v31, p_in_3, input_depth);
          vld_b_sp_xx(v32, p_in_3, input_depth);
          vld_b_sp_xx(v33, p_in_3, input_depth);
          vld_b_sp_xx(v34, p_in_4, input_depth);
          vld_b_sp_xx(v35, p_in_4, input_depth);
          vld_b_sp_xx(v36, p_in_4, input_depth);
          vld_b_sp_xx(v37, p_in_4, input_depth);
          vld_b_sp_xx(v38, p_in_4, input_depth);
          vld_b_sp_xx(v39, p_in_4, input_depth);
          vld_b_sp_xx(v40, p_in_4, input_depth);
          vld_b_sp_xx(v41, p_in_4, input_depth);

          // Compute bottom 2x5, starting at 3,3
          adwinit_v(v60, v60);
          adwconv_vxv(v60, v29, cmds, FLT_3_0);
          adwconv_vxv(v60, v32, cmds, FLT_3_3);
          adwconv_vxv(v60, v36, cmds, FLT_HOLE);
          vdwconv_vxv(v60, v39, cmds, FLT_4_2);

          // Compute bottom 2x5, starting at 3,2
          adwinit_v(v56, v56);
          adwconv_vxv(v56, v28, cmds, FLT_3_0);
          adwconv_vxv(v56, v31, cmds, FLT_3_3);
          adwconv_vxv(v56, v35, cmds, FLT_HOLE);
          vdwconv_vxv(v56, v38, cmds, FLT_4_2);

          // Compute bottom 2x5, starting at 3,1
          adwinit_v(v52, v52);
          adwconv_vxv(v52, v27, cmds, FLT_3_0);
          adwconv_vxv(v52, v30, cmds, FLT_3_3);
          adwconv_vxv(v52, v34, cmds, FLT_HOLE);
          vdwconv_vxv(v52, v37, cmds, FLT_4_2);

          // Compute bottom 2x5, starting at 3,0
          adwinit_v(v48, v48);
          adwconv_vxv(v48, v26, cmds, FLT_3_0);
          adwconv_vxv(v48, v29, cmds, FLT_3_3);
          adwconv_vxv(v48, v33, cmds, FLT_HOLE);
          vdwconv_vxv(v48, v36, cmds, FLT_4_2);

          // Load output parameters
          vld_w_x_m(v40, swizzled_output_multi);
          vld_w_x_m(v44, swizzled_shift_multi);
          vrsub_w_vx_m(v44, v44, 0);

          // Compute final outputs, for both 5x5 patches, and store.
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE4(v60, v56, v52, v48, v40, v44, output_activation_min, output_activation_max, output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);
          p_output += output_depth;
          vsraqs_b_vx(v52, v52, 0);
          vst_b_x(v52, p_output);
          p_output += output_depth;
          vsraqs_b_vx(v56, v56, 0);
          vst_b_x(v56, p_output);
          p_output += output_depth;
          vsraqs_b_vx(v60, v60, 0);
          vst_b_x(v60, p_output);
          p_output += output_depth;
        }
        // These were clobbered due to the different compute pattern
        // in the previous loop, so re-load them.
        vld_w_x_m(v56, swizzled_output_multi);
        vld_w_x_m(v52, swizzled_shift_multi);
        vrsub_w_vx_m(v52, v52, 0);
        for (; out_x < output_width - pad_width; ++out_x) {
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_4, p_in_0, input_depth);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, input_depth);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          vld_b_sp_xx(INPUT_3_0, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_in_3, input_depth);

          vld_b_sp_xx(INPUT_4_0, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_1, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_4, p_in_4, input_depth);

          COMPUTE();
          p_output += output_depth;
        }
        { // out_x == output_width - 2
          INPUT_PTRS(0);
          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_in_0, input_depth);
          vdup_b_x(INPUT_0_4, -input_offset);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vdup_b_x(INPUT_1_4, -input_offset);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vdup_b_x(INPUT_2_4, -input_offset);

          vld_b_sp_xx(INPUT_3_0, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vdup_b_x(INPUT_3_4, -input_offset);

          vld_b_sp_xx(INPUT_4_0, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_1, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_in_4, input_depth);
          vdup_b_x(INPUT_4_4, -input_offset);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        { // out_x == output_width - 1
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vdup_b_x(INPUT_0_3, -input_offset);
          vdup_b_x(INPUT_0_4, -input_offset);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vdup_b_x(INPUT_1_3, -input_offset);
          vdup_b_x(INPUT_1_4, -input_offset);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vdup_b_x(INPUT_2_3, -input_offset);
          vdup_b_x(INPUT_2_4, -input_offset);

          vld_b_sp_xx(INPUT_3_0, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vdup_b_x(INPUT_3_3, -input_offset);
          vdup_b_x(INPUT_3_4, -input_offset);

          vld_b_sp_xx(INPUT_4_0, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_1, p_in_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_in_4, input_depth);
          vdup_b_x(INPUT_4_3, -input_offset);
          vdup_b_x(INPUT_4_4, -input_offset);

          COMPUTE();
          p_output += output_depth;
        }
      }
      // Done
      { // out_y == output_height - 2
        int out_x = 0;
        vdup_b_x(INPUT_4_0, -input_offset);
        vdup_b_x(INPUT_4_1, -input_offset);
        vdup_b_x(INPUT_4_2, -input_offset);
        vdup_b_x(INPUT_4_3, -input_offset);
        vdup_b_x(INPUT_4_4, -input_offset);
        { // out_x == 0
          INPUT_PTRS(2);

          vdup_b_x(INPUT_0_0, -input_offset);
          vdup_b_x(INPUT_0_1, -input_offset);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_4, p_in_0, input_depth);

          vdup_b_x(INPUT_1_0, -input_offset);
          vdup_b_x(INPUT_1_1, -input_offset);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, input_depth);

          vdup_b_x(INPUT_2_0, -input_offset);
          vdup_b_x(INPUT_2_1, -input_offset);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          vdup_b_x(INPUT_3_0, -input_offset);
          vdup_b_x(INPUT_3_1, -input_offset);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_in_3, input_depth);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        { // out_x == 1
          INPUT_PTRS(1);

          vdup_b_x(INPUT_0_0, -input_offset);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_4, p_in_0, input_depth);

          vdup_b_x(INPUT_1_0, -input_offset);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, input_depth);

          vdup_b_x(INPUT_2_0, -input_offset);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          vdup_b_x(INPUT_3_0, -input_offset);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_in_3, input_depth);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        for (; out_x < output_width - pad_width; ++out_x) {
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_4, p_in_0, input_depth);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, input_depth);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          vld_b_sp_xx(INPUT_3_0, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_in_3, input_depth);

          COMPUTE();
          p_output += output_depth;
        }
        { // out_x == output_width - 2
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_in_0, input_depth);
          vdup_b_x(INPUT_0_4, -input_offset);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vdup_b_x(INPUT_1_4, -input_offset);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vdup_b_x(INPUT_2_4, -input_offset);

          vld_b_sp_xx(INPUT_3_0, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_in_3, input_depth);
          vdup_b_x(INPUT_3_4, -input_offset);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        { // out_x == output_width - 1
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vdup_b_x(INPUT_0_3, -input_offset);
          vdup_b_x(INPUT_0_4, -input_offset);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vdup_b_x(INPUT_1_3, -input_offset);
          vdup_b_x(INPUT_1_4, -input_offset);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vdup_b_x(INPUT_2_3, -input_offset);
          vdup_b_x(INPUT_2_4, -input_offset);

          vld_b_sp_xx(INPUT_3_0, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_in_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_in_3, input_depth);
          vdup_b_x(INPUT_3_3, -input_offset);
          vdup_b_x(INPUT_3_4, -input_offset);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        ++out_y;
      }
      // Done
      { // out_y == output_height - 1
        int out_x = 0;
        vdup_b_x(INPUT_3_0, -input_offset);
        vdup_b_x(INPUT_3_1, -input_offset);
        vdup_b_x(INPUT_3_2, -input_offset);
        vdup_b_x(INPUT_3_3, -input_offset);
        vdup_b_x(INPUT_3_4, -input_offset);

        vdup_b_x(INPUT_4_0, -input_offset);
        vdup_b_x(INPUT_4_1, -input_offset);
        vdup_b_x(INPUT_4_2, -input_offset);
        vdup_b_x(INPUT_4_3, -input_offset);
        vdup_b_x(INPUT_4_4, -input_offset);
        { // out_x == 0
          INPUT_PTRS(2);

          vdup_b_x(INPUT_0_0, -input_offset);
          vdup_b_x(INPUT_0_1, -input_offset);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_4, p_in_0, input_depth);

          vdup_b_x(INPUT_1_0, -input_offset);
          vdup_b_x(INPUT_1_1, -input_offset);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, input_depth);

          vdup_b_x(INPUT_2_0, -input_offset);
          vdup_b_x(INPUT_2_1, -input_offset);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        { // out_x == 1
          INPUT_PTRS(1);

          vdup_b_x(INPUT_0_0, -input_offset);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_4, p_in_0, input_depth);

          vdup_b_x(INPUT_1_0, -input_offset);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, input_depth);

          vdup_b_x(INPUT_2_0, -input_offset);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        for (; out_x < output_width - pad_width; ++out_x) {
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_4, p_in_0, input_depth);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_in_1, input_depth);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_in_2, input_depth);

          COMPUTE();
          p_output += output_depth;
        }
        { // out_x == output_width - 2
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_in_0, input_depth);
          vdup_b_x(INPUT_0_4, -input_offset);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_in_1, input_depth);
          vdup_b_x(INPUT_1_4, -input_offset);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_in_2, input_depth);
          vdup_b_x(INPUT_2_4, -input_offset);

          COMPUTE();
          p_output += output_depth;
          ++out_x;
        }
        { // out_x == output_width - 1
          INPUT_PTRS(0);

          vld_b_sp_xx(INPUT_0_0, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_in_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_in_0, input_depth);
          vdup_b_x(INPUT_0_3, -input_offset);
          vdup_b_x(INPUT_0_4, -input_offset);

          vld_b_sp_xx(INPUT_1_0, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_in_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_in_1, input_depth);
          vdup_b_x(INPUT_1_3, -input_offset);
          vdup_b_x(INPUT_1_4, -input_offset);

          vld_b_sp_xx(INPUT_2_0, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_in_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_in_2, input_depth);
          vdup_b_x(INPUT_2_3, -input_offset);
          vdup_b_x(INPUT_2_4, -input_offset);

          COMPUTE();
          p_output += output_depth;
        }
      }
    }
  }
#undef INPUT_PTRS
#undef COMPUTE
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
}

// special case of input depth = 32n, filter shape of 5x5
void DepthwiseConvS85x5D32(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data
) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);
  int32_t swizzled_bias_data[32];
  int32_t swizzled_shift_multi[32];
  int32_t swizzled_output_multi[32];

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

  for (int in_channel = 0; in_channel + 32 <= input_depth; in_channel += 32) {
    const int output_channel = in_channel;
    VectorSwizzle(bias_data + output_channel, swizzled_bias_data, 32);
    VectorSwizzle(output_multiplier + output_channel, swizzled_output_multi, 32);
    VectorSwizzle(output_shift + output_channel, swizzled_shift_multi, 32);

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

    vld_w_x_m(v56, swizzled_output_multi);
    vld_w_x_m(v60, swizzled_shift_multi);
    vrsub_w_vx_m(v60, v60, 0);

    const int8_t* p_flt = filter_data + in_channel;
    vld_b_sp_xx(FLT_0_0, p_flt, input_depth);
    vld_b_sp_xx(FLT_0_1, p_flt, input_depth);
    vld_b_sp_xx(FLT_0_2, p_flt, input_depth);
    vld_b_sp_xx(FLT_0_3, p_flt, input_depth);
    vld_b_sp_xx(FLT_0_4, p_flt, input_depth);

    vld_b_sp_xx(FLT_1_0, p_flt, input_depth);
    vld_b_sp_xx(FLT_1_1, p_flt, input_depth);
    vld_b_sp_xx(FLT_1_2, p_flt, input_depth);
    vld_b_sp_xx(FLT_1_3, p_flt, input_depth);
    vld_b_sp_xx(FLT_1_4, p_flt, input_depth);

    vld_b_sp_xx(FLT_2_0, p_flt, input_depth);
    vld_b_sp_xx(FLT_2_1, p_flt, input_depth);
    vld_b_sp_xx(FLT_2_2, p_flt, input_depth);
    vld_b_sp_xx(FLT_2_3, p_flt, input_depth);
    vld_b_sp_xx(FLT_2_4, p_flt, input_depth);

    vld_b_sp_xx(FLT_3_0, p_flt, input_depth);
    vld_b_sp_xx(FLT_3_1, p_flt, input_depth);
    vld_b_sp_xx(FLT_3_2, p_flt, input_depth);
    vld_b_sp_xx(FLT_3_3, p_flt, input_depth);
    vld_b_sp_xx(FLT_3_4, p_flt, input_depth);

    vld_b_sp_xx(FLT_4_0, p_flt, input_depth);
    vld_b_sp_xx(FLT_4_1, p_flt, input_depth);
    vld_b_sp_xx(FLT_4_2, p_flt, input_depth);
    vld_b_sp_xx(FLT_4_3, p_flt, input_depth);
    vld_b_sp_xx(FLT_4_4, p_flt, input_depth);
    vdup_b_x(FLT_HOLE, 0);

#define COMPUTE()                              \
  vld_w_x_m(v52, swizzled_bias_data);          \
  adwinit_v(v52, v52);                         \
  adwconv_vxv(v52, INPUT_0_0, cmds, FLT_0_0);  \
  adwconv_vxv(v52, INPUT_0_1, cmds, FLT_0_1);  \
  adwconv_vxv(v52, INPUT_0_2, cmds, FLT_0_2);  \
  adwconv_vxv(v52, INPUT_0_3, cmds, FLT_0_3);  \
  adwconv_vxv(v52, INPUT_0_4, cmds, FLT_0_4);  \
  adwconv_vxv(v52, INPUT_3_0, cmds, FLT_3_0);  \
  adwconv_vxv(v52, INPUT_3_3, cmds, FLT_3_3);  \
  adwconv_vxv(v52, INPUT_3_4, cmds, FLT_HOLE); \
  vdwconv_vxv(v52, INPUT_4_2, cmds, FLT_4_2);

#define INPUT_PTRS()                                                    \
  const int8_t* p_input_0 =                                             \
      input_data + (batch * input_height * input_width * input_depth) + \
      (in_y_origin * input_width * input_depth) +                       \
      (in_x_origin * input_depth) + in_channel;                         \
  const int8_t* p_input_1 = p_input_0 + (input_width * input_depth);    \
  const int8_t* p_input_2 = p_input_1 + (input_width * input_depth);    \
  const int8_t* p_input_3 = p_input_2 + (input_width * input_depth);    \
  const int8_t* p_input_4 = p_input_3 + (input_width * input_depth);    \
  (void)p_input_4;

    for (int batch = 0; batch < batches; ++batch) {
      int out_y = 0;
      int8_t* p_output = output_data +
                         (batch * output_height * output_width * output_depth) +
                         output_channel;
      do {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        if (in_y_origin >= 0) {
          break;
        }
        int out_x = 0;
        do {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          if (in_x_origin >= 0) {
            break;
          }
          INPUT_PTRS();
#define LOAD_INPUT(y, x)                                       \
  if (in_y_origin + y < 0) {                                   \
    vdup_b_x(INPUT_##y##_##x, -input_offset);                  \
  } else if (in_x_origin + x < 0) {                            \
    vdup_b_x(INPUT_##y##_##x, -input_offset);                  \
  } else {                                                     \
    vld_b_x(INPUT_##y##_##x, p_input_##y + (x * input_depth)); \
  }

          LOAD_INPUT(0, 0);
          LOAD_INPUT(0, 1);
          LOAD_INPUT(0, 2);
          LOAD_INPUT(0, 3);
          LOAD_INPUT(0, 4);
          LOAD_INPUT(1, 0);
          LOAD_INPUT(1, 1);
          LOAD_INPUT(1, 2);
          LOAD_INPUT(1, 3);
          LOAD_INPUT(1, 4);
          LOAD_INPUT(2, 0);
          LOAD_INPUT(2, 1);
          LOAD_INPUT(2, 2);
          LOAD_INPUT(2, 3);
          LOAD_INPUT(2, 4);
          LOAD_INPUT(3, 0);
          LOAD_INPUT(3, 1);
          LOAD_INPUT(3, 2);
          LOAD_INPUT(3, 3);
          LOAD_INPUT(3, 4);
          LOAD_INPUT(4, 0);
          LOAD_INPUT(4, 1);
          LOAD_INPUT(4, 2);
          LOAD_INPUT(4, 3);
          LOAD_INPUT(4, 4);
#undef LOAD_INPUT
          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v52, v56, v60, output_activation_min, output_activation_max,
              output_offset);
          vsraqs_b_vx(v52, v52, 0);
          vst_b_x(v52, p_output);
          p_output += output_depth;
          ++out_x;
        } while (out_x < output_width);
        do {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          if (in_x_origin + 4 >= input_width) {
            break;
          }
          INPUT_PTRS();
          vdup_b_x(INPUT_0_0, -input_offset);
          vdup_b_x(INPUT_0_1, -input_offset);
          vdup_b_x(INPUT_0_2, -input_offset);
          vdup_b_x(INPUT_0_3, -input_offset);
          vdup_b_x(INPUT_0_4, -input_offset);
          if (in_y_origin + 1 < 0) {
            vdup_b_x(INPUT_1_0, -input_offset);
            vdup_b_x(INPUT_1_1, -input_offset);
            vdup_b_x(INPUT_1_2, -input_offset);
            vdup_b_x(INPUT_1_3, -input_offset);
            vdup_b_x(INPUT_1_4, -input_offset);
          } else {
            vld_b_sp_xx(INPUT_1_0, p_input_1, input_depth);
            vld_b_sp_xx(INPUT_1_1, p_input_1, input_depth);
            vld_b_sp_xx(INPUT_1_2, p_input_1, input_depth);
            vld_b_sp_xx(INPUT_1_3, p_input_1, input_depth);
            vld_b_sp_xx(INPUT_1_4, p_input_1, input_depth);
          }
          if (in_y_origin + 2 < 0) {
            vdup_b_x(INPUT_2_0, -input_offset);
            vdup_b_x(INPUT_2_1, -input_offset);
            vdup_b_x(INPUT_2_2, -input_offset);
            vdup_b_x(INPUT_2_3, -input_offset);
            vdup_b_x(INPUT_2_4, -input_offset);
          } else {
            vld_b_sp_xx(INPUT_2_0, p_input_2, input_depth);
            vld_b_sp_xx(INPUT_2_1, p_input_2, input_depth);
            vld_b_sp_xx(INPUT_2_2, p_input_2, input_depth);
            vld_b_sp_xx(INPUT_2_3, p_input_2, input_depth);
            vld_b_sp_xx(INPUT_2_4, p_input_2, input_depth);
          }
          if (in_y_origin + 3 < 0) {
            vdup_b_x(INPUT_3_0, -input_offset);
            vdup_b_x(INPUT_3_1, -input_offset);
            vdup_b_x(INPUT_3_2, -input_offset);
            vdup_b_x(INPUT_3_3, -input_offset);
            vdup_b_x(INPUT_3_4, -input_offset);
          } else {
            vld_b_sp_xx(INPUT_3_0, p_input_3, input_depth);
            vld_b_sp_xx(INPUT_3_1, p_input_3, input_depth);
            vld_b_sp_xx(INPUT_3_2, p_input_3, input_depth);
            vld_b_sp_xx(INPUT_3_3, p_input_3, input_depth);
            vld_b_sp_xx(INPUT_3_4, p_input_3, input_depth);
          }
          if (in_y_origin + 4 < 0) {
            vdup_b_x(INPUT_4_0, -input_offset);
            vdup_b_x(INPUT_4_1, -input_offset);
            vdup_b_x(INPUT_4_2, -input_offset);
            vdup_b_x(INPUT_4_3, -input_offset);
            vdup_b_x(INPUT_4_4, -input_offset);
          } else {
            vld_b_sp_xx(INPUT_4_0, p_input_4, input_depth);
            vld_b_sp_xx(INPUT_4_1, p_input_4, input_depth);
            vld_b_sp_xx(INPUT_4_2, p_input_4, input_depth);
            vld_b_sp_xx(INPUT_4_3, p_input_4, input_depth);
            vld_b_sp_xx(INPUT_4_4, p_input_4, input_depth);
          }
          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v52, v56, v60, output_activation_min, output_activation_max,
              output_offset);
          vsraqs_b_vx(v52, v52, 0);
          vst_b_x(v52, p_output);
          p_output += output_depth;
          ++out_x;
        } while (out_x < output_width);
        do {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          INPUT_PTRS();
#define LOAD_INPUT(y, x)                                       \
  if (in_y_origin + y < 0) {                                   \
    vdup_b_x(INPUT_##y##_##x, -input_offset);                  \
  } else if (in_x_origin + x >= input_width) {                 \
    vdup_b_x(INPUT_##y##_##x, -input_offset);                  \
  } else {                                                     \
    vld_b_x(INPUT_##y##_##x, p_input_##y + (x * input_depth)); \
  }

          LOAD_INPUT(0, 0);
          LOAD_INPUT(0, 1);
          LOAD_INPUT(0, 2);
          LOAD_INPUT(0, 3);
          LOAD_INPUT(0, 4);
          LOAD_INPUT(1, 0);
          LOAD_INPUT(1, 1);
          LOAD_INPUT(1, 2);
          LOAD_INPUT(1, 3);
          LOAD_INPUT(1, 4);
          LOAD_INPUT(2, 0);
          LOAD_INPUT(2, 1);
          LOAD_INPUT(2, 2);
          LOAD_INPUT(2, 3);
          LOAD_INPUT(2, 4);
          LOAD_INPUT(3, 0);
          LOAD_INPUT(3, 1);
          LOAD_INPUT(3, 2);
          LOAD_INPUT(3, 3);
          LOAD_INPUT(3, 4);
          LOAD_INPUT(4, 0);
          LOAD_INPUT(4, 1);
          LOAD_INPUT(4, 2);
          LOAD_INPUT(4, 3);
          LOAD_INPUT(4, 4);
#undef LOAD_INPUT
          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v52, v56, v60, output_activation_min, output_activation_max,
              output_offset);
          vsraqs_b_vx(v52, v52, 0);
          vst_b_x(v52, p_output);
          p_output += output_depth;
          ++out_x;
        } while (out_x < output_width);
        ++out_y;
      } while (out_y < output_height);
      do {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        if (in_y_origin + 4 >= input_height) {
          break;
        }
        int out_x = 0;
        do {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          if (in_x_origin >= 0) {
            break;
          }
          INPUT_PTRS();
          vdup_b_x(INPUT_0_0, -input_offset);
          vdup_b_x(INPUT_1_0, -input_offset);
          vdup_b_x(INPUT_2_0, -input_offset);
          vdup_b_x(INPUT_3_0, -input_offset);
          vdup_b_x(INPUT_4_0, -input_offset);
          if (in_x_origin + 1 < 0) {
            vdup_b_x(INPUT_0_1, -input_offset);
            vdup_b_x(INPUT_1_1, -input_offset);
            vdup_b_x(INPUT_2_1, -input_offset);
            vdup_b_x(INPUT_3_1, -input_offset);
            vdup_b_x(INPUT_4_1, -input_offset);
          } else {
            vld_b_x(INPUT_0_1, p_input_0 + (1 * input_depth));
            vld_b_x(INPUT_1_1, p_input_1 + (1 * input_depth));
            vld_b_x(INPUT_2_1, p_input_2 + (1 * input_depth));
            vld_b_x(INPUT_3_1, p_input_3 + (1 * input_depth));
            vld_b_x(INPUT_4_1, p_input_4 + (1 * input_depth));
          }
          if (in_x_origin + 2 < 0) {
            vdup_b_x(INPUT_0_2, -input_offset);
            vdup_b_x(INPUT_1_2, -input_offset);
            vdup_b_x(INPUT_2_2, -input_offset);
            vdup_b_x(INPUT_3_2, -input_offset);
            vdup_b_x(INPUT_4_2, -input_offset);
          } else {
            vld_b_x(INPUT_0_2, p_input_0 + (2 * input_depth));
            vld_b_x(INPUT_1_2, p_input_1 + (2 * input_depth));
            vld_b_x(INPUT_2_2, p_input_2 + (2 * input_depth));
            vld_b_x(INPUT_3_2, p_input_3 + (2 * input_depth));
            vld_b_x(INPUT_4_2, p_input_4 + (2 * input_depth));
          }
          if (in_x_origin + 3 < 0) {
            vdup_b_x(INPUT_0_3, -input_offset);
            vdup_b_x(INPUT_1_3, -input_offset);
            vdup_b_x(INPUT_2_3, -input_offset);
            vdup_b_x(INPUT_3_3, -input_offset);
            vdup_b_x(INPUT_4_3, -input_offset);
          } else {
            vld_b_x(INPUT_0_3, p_input_0 + (3 * input_depth));
            vld_b_x(INPUT_1_3, p_input_1 + (3 * input_depth));
            vld_b_x(INPUT_2_3, p_input_2 + (3 * input_depth));
            vld_b_x(INPUT_3_3, p_input_3 + (3 * input_depth));
            vld_b_x(INPUT_4_3, p_input_4 + (3 * input_depth));
          }
          if (in_x_origin + 4 < 0) {
            vdup_b_x(INPUT_0_4, -input_offset);
            vdup_b_x(INPUT_1_4, -input_offset);
            vdup_b_x(INPUT_2_4, -input_offset);
            vdup_b_x(INPUT_3_4, -input_offset);
            vdup_b_x(INPUT_4_4, -input_offset);
          } else {
            vld_b_x(INPUT_0_4, p_input_0 + (4 * input_depth));
            vld_b_x(INPUT_1_4, p_input_1 + (4 * input_depth));
            vld_b_x(INPUT_2_4, p_input_2 + (4 * input_depth));
            vld_b_x(INPUT_3_4, p_input_3 + (4 * input_depth));
            vld_b_x(INPUT_4_4, p_input_4 + (4 * input_depth));
          }

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v52, v56, v60, output_activation_min, output_activation_max,
              output_offset);
          vsraqs_b_vx(v52, v52, 0);
          vst_b_x(v52, p_output);
          p_output += output_depth;
          ++out_x;
        } while (out_x < output_width);
        do {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          if (in_x_origin + 4 >= input_width) {
            break;
          }
          INPUT_PTRS();
          vld_b_sp_xx(INPUT_0_0, p_input_0, input_depth);
          vld_b_sp_xx(INPUT_0_1, p_input_0, input_depth);
          vld_b_sp_xx(INPUT_0_2, p_input_0, input_depth);
          vld_b_sp_xx(INPUT_0_3, p_input_0, input_depth);
          vld_b_sp_xx(INPUT_0_4, p_input_0, input_depth);
          vld_b_sp_xx(INPUT_1_0, p_input_1, input_depth);
          vld_b_sp_xx(INPUT_1_1, p_input_1, input_depth);
          vld_b_sp_xx(INPUT_1_2, p_input_1, input_depth);
          vld_b_sp_xx(INPUT_1_3, p_input_1, input_depth);
          vld_b_sp_xx(INPUT_1_4, p_input_1, input_depth);
          vld_b_sp_xx(INPUT_2_0, p_input_2, input_depth);
          vld_b_sp_xx(INPUT_2_1, p_input_2, input_depth);
          vld_b_sp_xx(INPUT_2_2, p_input_2, input_depth);
          vld_b_sp_xx(INPUT_2_3, p_input_2, input_depth);
          vld_b_sp_xx(INPUT_2_4, p_input_2, input_depth);
          vld_b_sp_xx(INPUT_3_0, p_input_3, input_depth);
          vld_b_sp_xx(INPUT_3_1, p_input_3, input_depth);
          vld_b_sp_xx(INPUT_3_2, p_input_3, input_depth);
          vld_b_sp_xx(INPUT_3_3, p_input_3, input_depth);
          vld_b_sp_xx(INPUT_3_4, p_input_3, input_depth);
          vld_b_sp_xx(INPUT_4_0, p_input_4, input_depth);
          vld_b_sp_xx(INPUT_4_1, p_input_4, input_depth);
          vld_b_sp_xx(INPUT_4_2, p_input_4, input_depth);
          vld_b_sp_xx(INPUT_4_3, p_input_4, input_depth);
          vld_b_sp_xx(INPUT_4_4, p_input_4, input_depth);

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v52, v56, v60, output_activation_min, output_activation_max,
              output_offset);
          vsraqs_b_vx(v52, v52, 0);
          vst_b_x(v52, p_output);
          p_output += output_depth;
          ++out_x;
        } while (out_x < output_width);
        do {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          INPUT_PTRS();
          if (in_x_origin >= input_width) {
            vdup_b_x(INPUT_0_0, -input_offset);
            vdup_b_x(INPUT_1_0, -input_offset);
            vdup_b_x(INPUT_2_0, -input_offset);
            vdup_b_x(INPUT_3_0, -input_offset);
            vdup_b_x(INPUT_4_0, -input_offset);
          } else {
            vld_b_x(INPUT_0_0, p_input_0);
            vld_b_x(INPUT_1_0, p_input_1);
            vld_b_x(INPUT_2_0, p_input_2);
            vld_b_x(INPUT_3_0, p_input_3);
            vld_b_x(INPUT_4_0, p_input_4);
          }
          if (in_x_origin + 1 >= input_width) {
            vdup_b_x(INPUT_0_1, -input_offset);
            vdup_b_x(INPUT_1_1, -input_offset);
            vdup_b_x(INPUT_2_1, -input_offset);
            vdup_b_x(INPUT_3_1, -input_offset);
            vdup_b_x(INPUT_4_1, -input_offset);
          } else {
            vld_b_x(INPUT_0_1, p_input_0 + (1 * input_depth));
            vld_b_x(INPUT_1_1, p_input_1 + (1 * input_depth));
            vld_b_x(INPUT_2_1, p_input_2 + (1 * input_depth));
            vld_b_x(INPUT_3_1, p_input_3 + (1 * input_depth));
            vld_b_x(INPUT_4_1, p_input_4 + (1 * input_depth));
          }
          if (in_x_origin + 2 >= input_width) {
            vdup_b_x(INPUT_0_2, -input_offset);
            vdup_b_x(INPUT_1_2, -input_offset);
            vdup_b_x(INPUT_2_2, -input_offset);
            vdup_b_x(INPUT_3_2, -input_offset);
            vdup_b_x(INPUT_4_2, -input_offset);
          } else {
            vld_b_x(INPUT_0_2, p_input_0 + (2 * input_depth));
            vld_b_x(INPUT_1_2, p_input_1 + (2 * input_depth));
            vld_b_x(INPUT_2_2, p_input_2 + (2 * input_depth));
            vld_b_x(INPUT_3_2, p_input_3 + (2 * input_depth));
            vld_b_x(INPUT_4_2, p_input_4 + (2 * input_depth));
          }
          if (in_x_origin + 3 >= input_width) {
            vdup_b_x(INPUT_0_3, -input_offset);
            vdup_b_x(INPUT_1_3, -input_offset);
            vdup_b_x(INPUT_2_3, -input_offset);
            vdup_b_x(INPUT_3_3, -input_offset);
            vdup_b_x(INPUT_4_3, -input_offset);
          } else {
            vld_b_x(INPUT_0_3, p_input_0 + (3 * input_depth));
            vld_b_x(INPUT_1_3, p_input_1 + (3 * input_depth));
            vld_b_x(INPUT_2_3, p_input_2 + (3 * input_depth));
            vld_b_x(INPUT_3_3, p_input_3 + (3 * input_depth));
            vld_b_x(INPUT_4_3, p_input_4 + (3 * input_depth));
          }
          vdup_b_x(INPUT_0_4, -input_offset);
          vdup_b_x(INPUT_1_4, -input_offset);
          vdup_b_x(INPUT_2_4, -input_offset);
          vdup_b_x(INPUT_3_4, -input_offset);
          vdup_b_x(INPUT_4_4, -input_offset);

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v52, v56, v60, output_activation_min, output_activation_max,
              output_offset);
          vsraqs_b_vx(v52, v52, 0);
          vst_b_x(v52, p_output);
          p_output += output_depth;
          ++out_x;
        } while (out_x < output_width);
        ++out_y;
      } while (out_y < output_height);
      do {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        int out_x = 0;
        do {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          if (in_x_origin >= 0) {
            break;
          }
          INPUT_PTRS();
#define LOAD_INPUT(y, x)                                       \
  if (in_y_origin + y >= input_height) {                       \
    vdup_b_x(INPUT_##y##_##x, -input_offset);                  \
  } else if (in_x_origin + x < 0) {                            \
    vdup_b_x(INPUT_##y##_##x, -input_offset);                  \
  } else {                                                     \
    vld_b_x(INPUT_##y##_##x, p_input_##y + (x * input_depth)); \
  }

          LOAD_INPUT(0, 0);
          LOAD_INPUT(0, 1);
          LOAD_INPUT(0, 2);
          LOAD_INPUT(0, 3);
          LOAD_INPUT(0, 4);
          LOAD_INPUT(1, 0);
          LOAD_INPUT(1, 1);
          LOAD_INPUT(1, 2);
          LOAD_INPUT(1, 3);
          LOAD_INPUT(1, 4);
          LOAD_INPUT(2, 0);
          LOAD_INPUT(2, 1);
          LOAD_INPUT(2, 2);
          LOAD_INPUT(2, 3);
          LOAD_INPUT(2, 4);
          LOAD_INPUT(3, 0);
          LOAD_INPUT(3, 1);
          LOAD_INPUT(3, 2);
          LOAD_INPUT(3, 3);
          LOAD_INPUT(3, 4);
          LOAD_INPUT(4, 0);
          LOAD_INPUT(4, 1);
          LOAD_INPUT(4, 2);
          LOAD_INPUT(4, 3);
          LOAD_INPUT(4, 4);
#undef LOAD_INPUT
          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v52, v56, v60, output_activation_min, output_activation_max,
              output_offset);
          vsraqs_b_vx(v52, v52, 0);
          vst_b_x(v52, p_output);
          p_output += output_depth;
          ++out_x;
        } while (out_x < output_width);
        do {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          if (in_x_origin + 4 >= input_width) {
            break;
          }
          INPUT_PTRS();
          if (in_y_origin >= input_height) {
            vdup_b_x(INPUT_0_0, -input_offset);
            vdup_b_x(INPUT_0_1, -input_offset);
            vdup_b_x(INPUT_0_2, -input_offset);
            vdup_b_x(INPUT_0_3, -input_offset);
            vdup_b_x(INPUT_0_4, -input_offset);
          } else {
            vld_b_sp_xx(INPUT_0_0, p_input_0, input_depth);
            vld_b_sp_xx(INPUT_0_1, p_input_0, input_depth);
            vld_b_sp_xx(INPUT_0_2, p_input_0, input_depth);
            vld_b_sp_xx(INPUT_0_3, p_input_0, input_depth);
            vld_b_sp_xx(INPUT_0_4, p_input_0, input_depth);
          }
          if (in_y_origin + 1 >= input_height) {
            vdup_b_x(INPUT_1_0, -input_offset);
            vdup_b_x(INPUT_1_1, -input_offset);
            vdup_b_x(INPUT_1_2, -input_offset);
            vdup_b_x(INPUT_1_3, -input_offset);
            vdup_b_x(INPUT_1_4, -input_offset);
          } else {
            vld_b_sp_xx(INPUT_1_0, p_input_1, input_depth);
            vld_b_sp_xx(INPUT_1_1, p_input_1, input_depth);
            vld_b_sp_xx(INPUT_1_2, p_input_1, input_depth);
            vld_b_sp_xx(INPUT_1_3, p_input_1, input_depth);
            vld_b_sp_xx(INPUT_1_4, p_input_1, input_depth);
          }
          if (in_y_origin + 2 >= input_height) {
            vdup_b_x(INPUT_2_0, -input_offset);
            vdup_b_x(INPUT_2_1, -input_offset);
            vdup_b_x(INPUT_2_2, -input_offset);
            vdup_b_x(INPUT_2_3, -input_offset);
            vdup_b_x(INPUT_2_4, -input_offset);
          } else {
            vld_b_sp_xx(INPUT_2_0, p_input_2, input_depth);
            vld_b_sp_xx(INPUT_2_1, p_input_2, input_depth);
            vld_b_sp_xx(INPUT_2_2, p_input_2, input_depth);
            vld_b_sp_xx(INPUT_2_3, p_input_2, input_depth);
            vld_b_sp_xx(INPUT_2_4, p_input_2, input_depth);
          }
          if (in_y_origin + 3 >= input_height) {
            vdup_b_x(INPUT_3_0, -input_offset);
            vdup_b_x(INPUT_3_1, -input_offset);
            vdup_b_x(INPUT_3_2, -input_offset);
            vdup_b_x(INPUT_3_3, -input_offset);
            vdup_b_x(INPUT_3_4, -input_offset);
          } else {
            vld_b_sp_xx(INPUT_3_0, p_input_3, input_depth);
            vld_b_sp_xx(INPUT_3_1, p_input_3, input_depth);
            vld_b_sp_xx(INPUT_3_2, p_input_3, input_depth);
            vld_b_sp_xx(INPUT_3_3, p_input_3, input_depth);
            vld_b_sp_xx(INPUT_3_4, p_input_3, input_depth);
          }
          vdup_b_x(INPUT_4_0, -input_offset);
          vdup_b_x(INPUT_4_1, -input_offset);
          vdup_b_x(INPUT_4_2, -input_offset);
          vdup_b_x(INPUT_4_3, -input_offset);
          vdup_b_x(INPUT_4_4, -input_offset);

          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v52, v56, v60, output_activation_min, output_activation_max,
              output_offset);
          vsraqs_b_vx(v52, v52, 0);
          vst_b_x(v52, p_output);
          p_output += output_depth;
          ++out_x;
        } while (out_x < output_width);
        do {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          INPUT_PTRS();
#define LOAD_INPUT(y, x)                                       \
  if (in_y_origin + y >= input_height) {                       \
    vdup_b_x(INPUT_##y##_##x, -input_offset);                  \
  } else if (in_x_origin + x >= input_width) {                 \
    vdup_b_x(INPUT_##y##_##x, -input_offset);                  \
  } else {                                                     \
    vld_b_x(INPUT_##y##_##x, p_input_##y + (x * input_depth)); \
  }

          LOAD_INPUT(0, 0);
          LOAD_INPUT(0, 1);
          LOAD_INPUT(0, 2);
          LOAD_INPUT(0, 3);
          LOAD_INPUT(0, 4);
          LOAD_INPUT(1, 0);
          LOAD_INPUT(1, 1);
          LOAD_INPUT(1, 2);
          LOAD_INPUT(1, 3);
          LOAD_INPUT(1, 4);
          LOAD_INPUT(2, 0);
          LOAD_INPUT(2, 1);
          LOAD_INPUT(2, 2);
          LOAD_INPUT(2, 3);
          LOAD_INPUT(2, 4);
          LOAD_INPUT(3, 0);
          LOAD_INPUT(3, 1);
          LOAD_INPUT(3, 2);
          LOAD_INPUT(3, 3);
          LOAD_INPUT(3, 4);
          LOAD_INPUT(4, 0);
          LOAD_INPUT(4, 1);
          LOAD_INPUT(4, 2);
          LOAD_INPUT(4, 3);
          LOAD_INPUT(4, 4);
#undef LOAD_INPUT
          COMPUTE();
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v52, v56, v60, output_activation_min, output_activation_max,
              output_offset);
          vsraqs_b_vx(v52, v52, 0);
          vst_b_x(v52, p_output);
          p_output += output_depth;
          ++out_x;
        } while (out_x < output_width);
        ++out_y;
      } while (out_y < output_height);
    }

#undef COMPUTE
#undef INPUT_PTRS
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
  }
}

// special case of input depth = 32n
void DepthwiseConvS8D32(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data

) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  int32_t swizzled_bias_data[32];
  int32_t swizzled_shift_multi[32];
  int32_t swizzled_output_multi[32];

  for (int in_channel = 0; in_channel + 32 <= input_depth; in_channel += 32) {
    const int output_channel = in_channel;
    VectorSwizzle(bias_data + output_channel, swizzled_bias_data, 32);
    VectorSwizzle(output_multiplier + output_channel, swizzled_output_multi, 32);
    VectorSwizzle(output_shift + output_channel, swizzled_shift_multi, 32);

    vld_w_x_m(v20, swizzled_bias_data);
    vld_w_x_m(v24, swizzled_output_multi);
    vld_w_x_m(v28, swizzled_shift_multi);
    vrsub_w_vx_m(v28, v28, 0);

    for (int batch = 0; batch < batches; ++batch) {
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;

          vdup_w_x_m(v48, 0);
          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + filter_y;
            if ((in_y < 0) || (in_y >= input_height)) {
              continue;
            }
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              if ((in_x < 0) || (in_x >= input_width)) {
                continue;
              }

              vld_b_x(v0, &input_data[tflite::Offset(input_shape, batch, in_y,
                                                     in_x, in_channel)]);  // xp
              vld_b_x(v4, &filter_data[tflite::Offset(filter_shape, 0, filter_y,
                                                      filter_x, in_channel)]);

              vaddw_h_vx(v0, v0, 0);
              vadd_h_vx(v0, v0, static_cast<int16_t>(input_offset));
              vadd_h_vx(v1, v1,
                        static_cast<int16_t>(input_offset));  // v0 v1 input

              vaddw_h_vx(v4, v4, static_cast<int16_t>(0));
              vmulw_w_vv(v8, v0, v4);
              vmulw_w_vv(v10, v1, v5);

              vadd_w_vv_m(v48, v48, v8);
            }
          }

          vadd_w_vv_m(v48, v48, v20);  // add bias
          vdmulh_w_rn_vv_m(v48, v48, v24);
          vsha_w_r_vv_m(v48, v48, v28);
          vadd_w_vx_m(v48, v48, output_offset);
          vmax_w_vx_m(v48, v48, output_activation_min);
          vmin_w_vx_m(v48, v48, output_activation_max);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, &output_data[tflite::Offset(output_shape, batch, out_y,
                                                   out_x, output_channel)]);
        }
      }
    }
  }
}

void DepthwiseConvS8D16(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data

) {
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t input_offset = params.input_offset;
  const int32_t output_offset = params.output_offset;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;
  const int batches = MatchingDim(input_shape, 0, output_shape, 0);
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  const int input_depth = input_shape.Dims(3);
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);
  const int output_depth = output_shape.Dims(3);
  for (int in_channel = 0; in_channel + 16 <= input_depth; in_channel += 16) {
    const int output_channel = in_channel;

    vld_w_x(v24, output_multiplier);
    vld_w_x(v25, output_multiplier + 8);
    vld_w_x(v28, output_shift);
    vld_w_x(v29, output_shift + 8);
    vrsub_w_vx(v28, v28, 0);
    vrsub_w_vx(v29, v29, 0);

    for (int batch = 0; batch < batches; ++batch) {
      const int8_t* p_output =
          output_data + (batch * output_width * output_height * output_depth) +
          output_channel;
      for (int out_y = 0; out_y < output_height; ++out_y) {
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;
          const int y_offset = (output_depth * output_width * out_y);

          if (bias_data) {
            vld_w_x(v48, bias_data);
            vld_w_x(v49, bias_data + 8);
          } else {
            vdup_w_x(v48, 0);
            vdup_w_x(v49, 0);
          }

          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + filter_y;
            if ((in_y < 0) || (in_y >= input_height)) {
              continue;
            }
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              if ((in_x < 0) || (in_x >= input_width)) {
                continue;
              }

              const int8_t* in_p =
                  input_data +
                  (batch * input_height * input_width * input_depth) +
                  (in_y * input_width * input_depth) + (in_x * input_depth) +
                  in_channel;

              const int8_t* fl_p = filter_data +
                                   (filter_y * filter_width * input_depth) +
                                   (filter_x * input_depth) + in_channel;

              vld_b_l_xx(v0, in_p, 16);
              vld_b_l_xx(v4, fl_p, 16);

              vaddw_h_vx(v0, v0, 0);
              vadd_h_vx(v0, v0, static_cast<int16_t>(input_offset));
              vadd_h_vx(v1, v1, static_cast<int16_t>(input_offset));
              vzip_h_vv(v0, v0, v1);

              vaddw_h_vx(v4, v4, static_cast<int16_t>(0));
              vzip_h_vv(v4, v4, v5);
              vmulw_w_vv(v8, v0, v4);

              vadd_w_vv(v48, v48, v8);
              vadd_w_vv(v49, v49, v9);
            }
          }

          vdmulh_w_rn_vv(v48, v48, v24);
          vdmulh_w_rn_vv(v49, v49, v25);
          vsha_w_r_vv(v48, v48, v28);
          vsha_w_r_vv(v49, v49, v29);

          vadd_w_vx(v48, v48, output_offset);
          vadd_w_vx(v49, v49, output_offset);
          vmax_w_vx(v48, v48, output_activation_min);
          vmax_w_vx(v49, v49, output_activation_min);
          vmin_w_vx(v48, v48, output_activation_max);
          vmin_w_vx(v49, v49, output_activation_max);
          vsraqs_b_vx_m(v48, v48, 0);
          vsraqs_b_vx(v49, v49, 0);
          vst_b_l_xx(v48, p_output + (out_x * output_depth) + y_offset, 16);
        }
      }
    }
  }
}

// generic implementation based on Kelvin ops
void DepthwiseConvS8Generic(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  // TBD: Use Kelvin implementation to replace the below
  tflite::reference_integer_ops::DepthwiseConvPerChannel(
      params, output_multiplier, output_shift, input_shape, input_data,
      filter_shape, filter_data, bias_shape, bias_data, output_shape,
      output_data);
  return;
}
}  // namespace

void DepthwiseConvS8(
    const tflite::DepthwiseParams& params, const int32_t* output_multiplier,
    const int32_t* output_shift, const tflite::RuntimeShape& input_shape,
    const int8_t* input_data, const tflite::RuntimeShape& filter_shape,
    const int8_t* filter_data, const tflite::RuntimeShape& bias_shape,
    const int32_t* bias_data, const tflite::RuntimeShape& output_shape,
    int8_t* output_data) {
  // Get parameters.
  // TODO(b/141565753): Re-introduce ScopedProfilingLabel on Micro.
  const int stride_width = params.stride_width;
  const int stride_height = params.stride_height;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
  const int dilation_width_factor = params.dilation_width_factor;
  const int dilation_height_factor = params.dilation_height_factor;
  const int depth_multiplier = params.depth_multiplier;
  const int32_t output_activation_min = params.quantized_activation_min;
  const int32_t output_activation_max = params.quantized_activation_max;

  // Check dimensions of the tensors.
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);

  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  const int output_depth = MatchingDim(filter_shape, 3, output_shape, 3);
  const int input_depth = input_shape.Dims(3);
  TFLITE_DCHECK_EQ(output_depth, input_depth * depth_multiplier);
  TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);

#define RUN_KERNEL(kernel) { \
  kernel(params, output_multiplier, output_shift, input_shape, input_data, \
       filter_shape, filter_data, bias_shape, bias_data, output_shape, \
       output_data \
  ); \
  return; \
}

  if (depth_multiplier == 1 &&
      dilation_height_factor == 1 && dilation_width_factor == 1 &&
      stride_height <= 2 && stride_width <= 2) {
    // special case of output depth = 32n
    if (output_depth % 32 == 0) {
      if (filter_width == 5 && filter_height == 5) {
        if (stride_width <= 1 && stride_height <= 1 && params.padding_type == tflite::PaddingType::kSame) {
          RUN_KERNEL(DepthwiseConvS85x5D32_Stride1);
        }
        RUN_KERNEL(DepthwiseConvS85x5D32);
      } if (filter_width == 3 && filter_height == 3 && pad_width <= 1 && pad_height <= 1 && stride_width == 1 && stride_height == 1) {
        RUN_KERNEL(DepthwiseConvS83x3D32_Stride1);
      } if (filter_width == 3 && filter_height == 3 && pad_width <= 1 && pad_height <= 1) {
        RUN_KERNEL(DepthwiseConvS83x3D32);
      }
      RUN_KERNEL(DepthwiseConvS8D32);
    } else if (output_depth % 16 == 0) {
      RUN_KERNEL(DepthwiseConvS8D16);
    }
    RUN_KERNEL(DepthwiseConvS8Generic);
  }

  RUN_KERNEL(tflite::reference_integer_ops::DepthwiseConvPerChannel);

#undef RUN_KERNEL
}

}  // namespace kelvin::opt
