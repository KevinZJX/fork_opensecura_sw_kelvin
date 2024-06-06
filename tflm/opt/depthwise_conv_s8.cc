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
      int out_y = 0;
      for (; out_y < pad_height; ++out_y) {
        int out_x = 0;
        const int in_y_origin = (out_y * stride_height) - pad_height;
        assert(in_y_origin < 0);
        vdup_b_x(v15, -input_offset);
        vdup_b_x(v16, -input_offset);
        vdup_b_x(v17, -input_offset);
        const int8_t* p_in_0 = input_data +
            (batch * input_height * input_width * input_depth) +
            (in_y_origin * input_width * input_depth) +
            (((out_x * stride_width) - pad_width) * input_depth) +
            in_channel;
        const int8_t* p_in_1 = p_in_0 + (input_width * input_depth);
        const int8_t* p_in_2 = p_in_1 + (input_width * input_depth);
        for (; out_x < pad_width; ++out_x) {
          vmv_v_m(v48, v52);

          vdup_b_x(v18, -input_offset);
          p_in_1 += input_depth;
          vld_b_sp_xx(v19, p_in_1, input_depth);
          vld_b_sp_xx(v20, p_in_1, input_depth);
          vdup_b_x(v21, -input_offset);
          p_in_2 += input_depth;
          vld_b_sp_xx(v22, p_in_2, input_depth);
          vld_b_sp_xx(v23, p_in_2, input_depth);

          adwinit_v(v48, v48);
          adwconv_vxv(v48, v15, cmds, v6);
          adwconv_vxv(v48, v18, cmds, v9);
          vdwconv_vxv(v48, v21, cmds, v12);

          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);

          p_output += output_depth;
          p_in_0 -= (2 * stride_width * input_depth);
          p_in_1 -= (2 * stride_width * input_depth);
          p_in_2 -= (2 * stride_width * input_depth);
        }
        for (; out_x < output_width - pad_width; ++out_x) {
          vmv_v_m(v48, v52);
          vld_b_sp_xx(v18, p_in_1, input_depth);
          vld_b_sp_xx(v19, p_in_1, input_depth);
          vld_b_sp_xx(v20, p_in_1, input_depth);
          vld_b_sp_xx(v21, p_in_2, input_depth);
          vld_b_sp_xx(v22, p_in_2, input_depth);
          vld_b_sp_xx(v23, p_in_2, input_depth);

          adwinit_v(v48, v48);
          adwconv_vxv(v48, v15, cmds, v6);
          adwconv_vxv(v48, v18, cmds, v9);
          vdwconv_vxv(v48, v21, cmds, v12);

          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);
          p_output += output_depth;
          p_in_0 -= (2 * stride_width * input_depth);
          p_in_1 -= (2 * stride_width * input_depth);
          p_in_2 -= (2 * stride_width * input_depth);
        }
        for (; out_x < output_width; ++out_x) {
          vmv_v_m(v48, v52);

          vld_b_sp_xx(v18, p_in_1, input_depth);
          vld_b_sp_xx(v19, p_in_1, input_depth);
          vdup_b_x(v20, -input_offset);
          vld_b_sp_xx(v21, p_in_2, input_depth);
          vld_b_sp_xx(v22, p_in_2, input_depth);
          vdup_b_x(v23, -input_offset);

          adwinit_v(v48, v48);
          adwconv_vxv(v48, v15, cmds, v6);
          adwconv_vxv(v48, v18, cmds, v9);
          vdwconv_vxv(v48, v21, cmds, v12);

          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);

          p_output += output_depth;
          p_in_0 -= (2 * stride_width * input_depth);
          p_in_1 -= (2 * stride_width * input_depth);
          p_in_2 -= (2 * stride_width * input_depth);
        }
      }
      for (; out_y < output_height - pad_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        int out_x = 0;
        const int8_t* p_in_0 = input_data +
            (batch * input_height * input_width * input_depth) +
            (in_y_origin * input_width * input_depth) +
            (((out_x * stride_width) - pad_width) * input_depth) +
            in_channel;
        const int8_t* p_in_1 = p_in_0 + (input_width * input_depth);
        const int8_t* p_in_2 = p_in_1 + (input_width * input_depth);
        for (; out_x < pad_width; ++out_x) {
          vmv_v_m(v48, v52);

          vdup_b_x(v15, -input_offset);
          vdup_b_x(v18, -input_offset);
          vdup_b_x(v21, -input_offset);
          p_in_0 += input_depth;
          p_in_1 += input_depth;
          p_in_2 += input_depth;
          vld_b_sp_xx(v16, p_in_0, input_depth);
          vld_b_sp_xx(v17, p_in_0, input_depth);
          vld_b_sp_xx(v19, p_in_1, input_depth);
          vld_b_sp_xx(v20, p_in_1, input_depth);
          vld_b_sp_xx(v22, p_in_2, input_depth);
          vld_b_sp_xx(v23, p_in_2, input_depth);

          adwinit_v(v48, v48);
          adwconv_vxv(v48, v15, cmds, v6);
          adwconv_vxv(v48, v18, cmds, v9);
          vdwconv_vxv(v48, v21, cmds, v12);
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);

          p_output += output_depth;
          p_in_0 -= (2 * stride_width * input_depth);
          p_in_1 -= (2 * stride_width * input_depth);
          p_in_2 -= (2 * stride_width * input_depth);
        }
        for (; out_x + 2 <= output_width - pad_width; out_x += 2) {
          // Initialize accumulators w/ bias data.
          vmv_v_m(v44, v52);
          vmv_v_m(v48, v52);

          vld_b_sp_xx(v15, p_in_0, stride_width * input_depth);
          vld_b_sp_xx(v16, p_in_0, stride_width * input_depth);
          vld_b_sp_xx(v17, p_in_0, stride_width * input_depth);
          vld_b_sp_xx(v18, p_in_0, stride_width * input_depth);
          vld_b_sp_xx(v19, p_in_1, stride_width * input_depth);
          vld_b_sp_xx(v20, p_in_1, stride_width * input_depth);
          vld_b_sp_xx(v21, p_in_1, stride_width * input_depth);
          vld_b_sp_xx(v22, p_in_1, stride_width * input_depth);
          vld_b_sp_xx(v23, p_in_2, stride_width * input_depth);
          vld_b_sp_xx(v24, p_in_2, stride_width * input_depth);
          vld_b_sp_xx(v25, p_in_2, stride_width * input_depth);
          vld_b_sp_xx(v26, p_in_2, stride_width * input_depth);

          adwinit_v(v48, v48);
          adwconv_vxv(v48, v15, cmds, v6);
          adwconv_vxv(v48, v19, cmds, v9);
          vdwconv_vxv(v48, v23, cmds, v12);

          adwinit_v(v44, v44);
          adwconv_vxv(v44, v16, cmds, v6);
          adwconv_vxv(v44, v20, cmds, v9);
          vdwconv_vxv(v44, v24, cmds, v12);

          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v44, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vsraqs_b_vx(v44, v44, 0);
          vst_b_x(v48, p_output);
          p_output += output_depth;
          vst_b_x(v44, p_output);
          p_output += output_depth;

          p_in_0 -= (2 * stride_width * input_depth);
          p_in_1 -= (2 * stride_width * input_depth);
          p_in_2 -= (2 * stride_width * input_depth);
        }
        for (; out_x < output_width - pad_width; ++out_x) {
          vmv_v_m(v48, v52);

          vld_b_sp_xx(v15, p_in_0, input_depth);
          vld_b_sp_xx(v16, p_in_0, input_depth);
          vld_b_sp_xx(v17, p_in_0, input_depth);
          vld_b_sp_xx(v18, p_in_1, input_depth);
          vld_b_sp_xx(v19, p_in_1, input_depth);
          vld_b_sp_xx(v20, p_in_1, input_depth);
          vld_b_sp_xx(v21, p_in_2, input_depth);
          vld_b_sp_xx(v22, p_in_2, input_depth);
          vld_b_sp_xx(v23, p_in_2, input_depth);

          adwinit_v(v48, v48);
          adwconv_vxv(v48, v15, cmds, v6);
          adwconv_vxv(v48, v18, cmds, v9);
          vdwconv_vxv(v48, v21, cmds, v12);

          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);
          p_output += output_depth;
          p_in_0 -= (2 * stride_width * input_depth);
          p_in_1 -= (2 * stride_width * input_depth);
          p_in_2 -= (2 * stride_width * input_depth);
        }
        for (; out_x < output_width; ++out_x) {
          vmv_v_m(v48, v52);

          vdup_b_x(v17, -input_offset);
          vdup_b_x(v20, -input_offset);
          vdup_b_x(v23, -input_offset);
          vld_b_sp_xx(v15, p_in_0, input_depth);
          vld_b_sp_xx(v16, p_in_0, input_depth);
          vld_b_sp_xx(v18, p_in_1, input_depth);
          vld_b_sp_xx(v19, p_in_1, input_depth);
          vld_b_sp_xx(v21, p_in_2, input_depth);
          vld_b_sp_xx(v22, p_in_2, input_depth);

          adwinit_v(v48, v48);
          adwconv_vxv(v48, v15, cmds, v6);
          adwconv_vxv(v48, v18, cmds, v9);
          vdwconv_vxv(v48, v21, cmds, v12);
          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);

          p_output += output_depth;
          p_in_0 -= (2 * stride_width * input_depth);
          p_in_1 -= (2 * stride_width * input_depth);
          p_in_2 -= (2 * stride_width * input_depth);
        }
      }
      for (; out_y < output_height; ++out_y) {
        const int in_y_origin = (out_y * stride_height) - pad_height;
        assert(in_y_origin + 2 >= input_height);
        vdup_b_x(v21, -input_offset);
        vdup_b_x(v22, -input_offset);
        vdup_b_x(v23, -input_offset);
        int out_x = 0;
        const int8_t* p_in_0 = input_data +
            (batch * input_height * input_width * input_depth) +
            (in_y_origin * input_width * input_depth) +
            (((out_x * stride_width) - pad_width) * input_depth) +
            in_channel;
        const int8_t* p_in_1 = p_in_0 + (input_width * input_depth);
        for (; out_x < pad_width; ++out_x) {
          vmv_v_m(v48, v52);

          vdup_b_x(v15, -input_offset);
          p_in_0 += input_depth;
          vld_b_sp_xx(v16, p_in_0, input_depth);
          vld_b_sp_xx(v17, p_in_0, input_depth);
          vdup_b_x(v18, -input_offset);
          p_in_1 += input_depth;
          vld_b_sp_xx(v19, p_in_1, input_depth);
          vld_b_sp_xx(v20, p_in_1, input_depth);

          adwinit_v(v48, v48);
          adwconv_vxv(v48, v15, cmds, v6);
          adwconv_vxv(v48, v18, cmds, v9);
          vdwconv_vxv(v48, v21, cmds, v12);

          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);
          p_output += output_depth;
          p_in_0 -= (2 * stride_width * input_depth);
          p_in_1 -= (2 * stride_width * input_depth);
        }
        for (; out_x < output_width - pad_width; ++out_x) {
          vmv_v_m(v48, v52);
          vld_b_sp_xx(v15, p_in_0, input_depth);
          vld_b_sp_xx(v16, p_in_0, input_depth);
          vld_b_sp_xx(v17, p_in_0, input_depth);
          vld_b_sp_xx(v18, p_in_1, input_depth);
          vld_b_sp_xx(v19, p_in_1, input_depth);
          vld_b_sp_xx(v20, p_in_1, input_depth);

          adwinit_v(v48, v48);
          adwconv_vxv(v48, v15, cmds, v6);
          adwconv_vxv(v48, v18, cmds, v9);
          vdwconv_vxv(v48, v21, cmds, v12);

          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);
          p_output += output_depth;
          p_in_0 -= (2 * stride_width * input_depth);
          p_in_1 -= (2 * stride_width * input_depth);
        }
        for (; out_x < output_width; ++out_x) {
          vmv_v_m(v48, v52);

          vld_b_sp_xx(v15, p_in_0, input_depth);
          vld_b_sp_xx(v16, p_in_0, input_depth);
          vdup_b_x(v17, -input_offset);
          vld_b_sp_xx(v18, p_in_1, input_depth);
          vld_b_sp_xx(v19, p_in_1, input_depth);
          vdup_b_x(v20, -input_offset);

          adwinit_v(v48, v48);
          adwconv_vxv(v48, v15, cmds, v6);
          adwconv_vxv(v48, v18, cmds, v9);
          vdwconv_vxv(v48, v21, cmds, v12);

          INT32_TO_INT8_OUTPUT_PIPELINE_INPLACE(
              v48, v56, v60,
              output_activation_min,
              output_activation_max,
              output_offset);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output);
          p_output += output_depth;
          p_in_0 -= (2 * stride_width * input_depth);
          p_in_1 -= (2 * stride_width * input_depth);
        }
      }
    }
  }
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
    vld_b_sp_xx_m(v0, p_flt0, stride);
    vld_b_sp_xx_m(v4, p_flt0, stride);
    vld_b_sp_xx_m(v8, p_flt0, stride);
    vld_b_sp_xx_m(v12, p_flt0, stride);
    vld_b_sp_xx_m(v16, p_flt0, stride);
    vld_b_sp_xx_m(v20, p_flt0, stride);
    vld_b_sp_xx(v24, p_flt0, stride);

    // Extra two registers to get our
    // total usage to a multiple of 3 for dwconv.
    vdup_b_x(v25, 0);
    vdup_b_x(v26, 0);

    for (int batch = 0; batch < batches; ++batch) {
      const int8_t* p_output = output_data + (batch * output_height * output_width * output_depth) + output_channel;
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int y_offset = out_y * output_width * output_depth;
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;

          bool top_pad = in_y_origin < 0;
          bool left_pad = in_x_origin < 0;
          int top_pad_count = top_pad ? 0 - in_y_origin : 0;
          int left_pad_count = left_pad ? 0 - in_x_origin : 0;
          bool bottom_pad = (in_y_origin + 4) >= input_height;
          bool right_pad = (in_x_origin + 4) >= input_width;
          int bottom_pad_count = std::abs(bottom_pad ? (in_y_origin + 4) - input_height + 1: 0);
          int right_pad_count = std::abs(right_pad ? (in_x_origin + 4) - input_width + 1 : 0);
          bool padding_required = top_pad || left_pad || bottom_pad || right_pad;
          assert(top_pad_count <= pad_height);
          assert(bottom_pad_count <= pad_height);
          assert(left_pad_count <= pad_width);
          assert(right_pad_count <= pad_width);
          assert(!(left_pad && right_pad));
          const int8_t* p_in_0 = input_data +
            (batch * input_height * input_width * input_depth) +
            (in_y_origin * input_width * input_depth) +
            (in_x_origin * input_depth) +
            in_channel;
          const int8_t* p_in_1 = p_in_0 + (input_width * input_depth);
          const int8_t* p_in_2 = p_in_1 + (input_width * input_depth);
          const int8_t* p_in_3 = p_in_2 + (input_width * input_depth);
          const int8_t* p_in_4 = p_in_3 + (input_width * input_depth);
          // Extra two registers to get our
          // total usage to a multiple of 3 for dwconv.
          vdup_b_x(v52, -input_offset);
          vdup_b_x(v53, -input_offset);
          if (!padding_required) {
            vld_b_sp_xx(v27, p_in_0, input_depth);
            vld_b_sp_xx_m(v28, p_in_0, input_depth);
            vld_b_sp_xx_m(v32, p_in_1, input_depth);
            vld_b_sp_xx(v36, p_in_1, input_depth);
            vld_b_sp_xx(v37, p_in_2, input_depth);
            vld_b_sp_xx(v38, p_in_2, input_depth);
            vld_b_sp_xx(v39, p_in_2, input_depth);
            vld_b_sp_xx(v40, p_in_2, input_depth);
            vld_b_sp_xx(v41, p_in_2, input_depth);
            vld_b_sp_xx(v42, p_in_3, input_depth);
            vld_b_sp_xx(v43, p_in_3, input_depth);
            vld_b_sp_xx(v44, p_in_3, input_depth);
            vld_b_sp_xx(v45, p_in_3, input_depth);
            vld_b_sp_xx(v46, p_in_3, input_depth);
            vld_b_sp_xx(v47, p_in_4, input_depth);
            vld_b_sp_xx_m(v48, p_in_4, input_depth);
          } else {
            // Top row
            if (top_pad_count >= 1) {
              vdup_b_x(v27, -input_offset);
              vdup_b_x_m(v28, -input_offset);
            } else {
              switch (left_pad_count) {
                case 2:
                  vdup_b_x(v28, -input_offset);
                case 1:
                  vdup_b_x(v27, -input_offset);
              }
              switch (left_pad_count) {
                case 0:
                  vld_b_x(v27, p_in_0);
                case 1:
                  vld_b_x(v28, p_in_0 + input_depth);
              }
              vld_b_x(v29, p_in_0 + (2 * input_depth));
              switch (right_pad_count) {
                case 2:
                  vdup_b_x(v30, -input_offset);
                case 1:
                  vdup_b_x(v31, -input_offset);
              }
              switch (right_pad_count) {
                case 0:
                  vld_b_x(v31, p_in_0 + (4 * input_depth));
                case 1:
                  vld_b_x(v30, p_in_0 + (3 * input_depth));
              }
            }

            // 2nd row
            if (top_pad_count == 2) {
              vdup_b_x_m(v32, -input_offset);
              vdup_b_x(v36, -input_offset);
            } else {
              switch (left_pad_count) {
                case 2:
                  vdup_b_x(v33, -input_offset);
                case 1:
                  vdup_b_x(v32, -input_offset);
              }
              switch (left_pad_count) {
                case 0:
                  vld_b_x(v32, p_in_1);
                case 1:
                  vld_b_x(v33, p_in_1 + input_depth);
              }
              vld_b_x(v34, p_in_1 + (2 * input_depth));
              switch (right_pad_count) {
                case 2:
                  vdup_b_x(v35, -input_offset);
                case 1:
                  vdup_b_x(v36, -input_offset);
              }
              switch (right_pad_count) {
                case 0:
                  vld_b_x(v36, p_in_1 + (4 * input_depth));
                case 1:
                  vld_b_x(v35, p_in_1 + (3 * input_depth));
              }
            }

            // 3rd row
            switch (left_pad_count) {
              case 2:
                vdup_b_x(v38, -input_offset);
              case 1:
                vdup_b_x(v37, -input_offset);
            }
            switch (left_pad_count) {
              case 0:
                vld_b_x(v37, p_in_2);
              case 1:
                vld_b_x(v38, p_in_2 + input_depth);
            }
            vld_b_x(v39, p_in_2 + (2 * input_depth));
            switch (right_pad_count) {
              case 2:
                vdup_b_x(v40, -input_offset);
              case 1:
                vdup_b_x(v41, -input_offset);
            }
            switch (right_pad_count) {
              case 0:
                vld_b_x(v41, p_in_2 + (4 * input_depth));
              case 1:
                vld_b_x(v40, p_in_2 + (3 * input_depth));
            }

            // 4th row
            if (bottom_pad_count == 2) {
              vdup_b_x(v42, -input_offset);
              vdup_b_x(v43, -input_offset);
              vdup_b_x(v44, -input_offset);
              vdup_b_x(v45, -input_offset);
              vdup_b_x(v46, -input_offset);
            } else {
              switch (left_pad_count) {
                case 2:
                  vdup_b_x(v43, -input_offset);
                case 1:
                  vdup_b_x(v42, -input_offset);
              }
              switch (left_pad_count) {
                case 0:
                  vld_b_x(v42, p_in_3);
                case 1:
                  vld_b_x(v43, p_in_3 + input_depth);
              }
              switch (right_pad_count) {
                case 2:
                  vdup_b_x(v45, -input_offset);
                case 1:
                  vdup_b_x(v46, -input_offset);
              }
              vld_b_x(v44, p_in_3 + (2 * input_depth));
              switch (right_pad_count) {
                case 0:
                  vld_b_x(v46, p_in_3 + (4 * input_depth));
                case 1:
                  vld_b_x(v45, p_in_3 + (3 * input_depth));
              }
            }

            // 5th row
            if (bottom_pad_count >= 1) {
              vdup_b_x(v47, -input_offset);
              vdup_b_x(v48, -input_offset);
              vdup_b_x(v49, -input_offset);
              vdup_b_x(v50, -input_offset);
              vdup_b_x(v51, -input_offset);
            } else {
              switch (left_pad_count) {
                case 2:
                  vdup_b_x(v48, -input_offset);
                case 1:
                  vdup_b_x(v47, -input_offset);
              }
              switch (left_pad_count) {
                case 0:
                  vld_b_x(v47, p_in_4);
                case 1:
                  vld_b_x(v48, p_in_4 + input_depth);
              }
              vld_b_x(v49, p_in_4 + (2 * input_depth));
              switch (right_pad_count) {
                case 2:
                  vdup_b_x(v50, -input_offset);
                case 1:
                  vdup_b_x(v51, -input_offset);
              }
              switch (right_pad_count) {
                case 0:
                  vld_b_x(v51, p_in_4 + (4 * input_depth));
                case 1:
                  vld_b_x(v50, p_in_4 + (3 * input_depth));
              }
            }
          }

          vld_w_x_m(v60, swizzled_bias_data);
          adwinit_v(v60, v60);
          adwconv_vxv(v60, v27, cmds, v0);
          adwconv_vxv(v60, v30, cmds, v3);
          adwconv_vxv(v60, v33, cmds, v6);
          adwconv_vxv(v60, v36, cmds, v9);
          adwconv_vxv(v60, v39, cmds, v12);
          adwconv_vxv(v60, v42, cmds, v15);
          adwconv_vxv(v60, v45, cmds, v18);
          adwconv_vxv(v60, v48, cmds, v21);
          vdwconv_vxv(v60, v51, cmds, v24);

          vld_w_x_m(v56, swizzled_output_multi);
          vdmulh_w_rn_vv_m(v60, v60, v56);
          vld_w_x_m(v56, swizzled_shift_multi);
          vrsub_w_vx_m(v56, v56, 0);
          vsha_w_r_vv_m(v60, v60, v56);
          vadd_w_vx_m(v60, v60, output_offset);
          vmax_w_vx_m(v60, v60, output_activation_min);
          vmin_w_vx_m(v60, v60, output_activation_max);
          vsraqs_b_vx(v60, v60, 0);
          vst_b_x(v60, p_output + y_offset + (out_x * output_depth));
        }
      }
    }
  }
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
  const int filter_height = filter_shape.Dims(1);
  const int filter_width = filter_shape.Dims(2);
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

    // Don't reorder me!
    const int8_t* p_flt = filter_data + in_channel;
    vld_b_sp_xx(v6, p_flt, input_depth);
    vld_b_sp_xx(v7, p_flt, input_depth);
    vld_b_sp_xx_m(v8, p_flt, input_depth);
    vld_b_sp_xx_m(v12, p_flt, input_depth);
    vld_b_sp_xx_m(v16, p_flt, input_depth);
    vld_b_sp_xx_m(v20, p_flt, input_depth);
    vld_b_sp_xx_m(v24, p_flt, input_depth);
    vld_b_sp_xx(v28, p_flt, input_depth);
    vld_b_sp_xx(v29, p_flt, input_depth);
    vld_b_sp_xx(v30, p_flt, input_depth);


    for (int batch = 0; batch < batches; ++batch) {
      const int8_t* p_input = input_data + (batch * input_width * input_height * input_depth) + in_channel;
      const int8_t* p_output = output_data + (batch * output_width * output_height * output_depth) + output_channel;
      for (int out_y = 0; out_y < output_height; ++out_y) {
        const int out_y_offset = (out_y * output_width * output_depth);
        for (int out_x = 0; out_x < output_width; ++out_x) {
          const int in_x_origin = (out_x * stride_width) - pad_width;
          const int in_y_origin = (out_y * stride_height) - pad_height;

          // Initialize accumulators w/ bias_data
          vmv_v_m(v48, v52);

          for (int filter_y = 0; filter_y < filter_height; ++filter_y) {
            const int in_y = in_y_origin + filter_y;
            if ((in_y < 0) || (in_y >= input_height)) {
              continue;
            }
            switch (filter_y) {
              case 0:
                vaddw_h_vx(v31, v6, 0);
                vaddw_h_vx(v33, v7, 0);
                vaddw_h_vx(v35, v8, 0);
                vaddw_h_vx(v37, v9, 0);
                vaddw_h_vx(v39, v10, 0);
                break;
              case 1:
                vaddw_h_vx(v31, v11, 0);
                vaddw_h_vx(v33, v12, 0);
                vaddw_h_vx(v35, v13, 0);
                vaddw_h_vx(v37, v14, 0);
                vaddw_h_vx(v39, v15, 0);
                break;
              case 2:
                vaddw_h_vx(v31, v16, 0);
                vaddw_h_vx(v33, v17, 0);
                vaddw_h_vx(v35, v18, 0);
                vaddw_h_vx(v37, v19, 0);
                vaddw_h_vx(v39, v20, 0);
                break;
              case 3:
                vaddw_h_vx(v31, v21, 0);
                vaddw_h_vx(v33, v22, 0);
                vaddw_h_vx(v35, v23, 0);
                vaddw_h_vx(v37, v24, 0);
                vaddw_h_vx(v39, v25, 0);
                break;
              case 4:
                vaddw_h_vx(v31, v26, 0);
                vaddw_h_vx(v33, v27, 0);
                vaddw_h_vx(v35, v28, 0);
                vaddw_h_vx(v37, v29, 0);
                vaddw_h_vx(v39, v30, 0);
                break;
            }
            const int in_y_offset = in_y  * input_width * input_depth;
            for (int filter_x = 0; filter_x < filter_width; ++filter_x) {
              const int in_x = in_x_origin + filter_x;
              if ((in_x < 0) || (in_x >= input_width)) {
                continue;
              }

              vld_b_x(v0, p_input + (in_x * input_depth) + in_y_offset);

              vaddw_h_vx(v0, v0, 0);
              vadd_h_vx(v0, v0, static_cast<int16_t>(input_offset));
              vadd_h_vx(v1, v1,
                        static_cast<int16_t>(input_offset));  // v0 v1 input
              switch (filter_x) {
                case 0:
                  vmulw_w_vv(v2, v1, v32);
                  vmulw_w_vv(v0, v0, v31);
                  break;
                case 1:
                  vmulw_w_vv(v2, v1, v34);
                  vmulw_w_vv(v0, v0, v33);
                  break;
                case 2:
                  vmulw_w_vv(v2, v1, v36);
                  vmulw_w_vv(v0, v0, v35);
                  break;
                case 3:
                  vmulw_w_vv(v2, v1, v38);
                  vmulw_w_vv(v0, v0, v37);
                  break;
                case 4:
                  vmulw_w_vv(v2, v1, v40);
                  vmulw_w_vv(v0, v0, v39);
                  break;
              }
              vadd_w_vv_m(v48, v48, v0);
            }
          }

          vdmulh_w_rn_vv_m(v48, v48, v56);
          vsha_w_r_vv_m(v48, v48, v60);
          vadd_w_vx_m(v48, v48, output_offset);
          vmax_w_vx_m(v48, v48, output_activation_min);
          vmin_w_vx_m(v48, v48, output_activation_max);
          vsraqs_b_vx(v48, v48, 0);
          vst_b_x(v48, p_output + out_y_offset + (out_x * output_depth));
        }
      }
    }
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
        if (stride_width <= 1 && stride_height <= 1) {
          RUN_KERNEL(DepthwiseConvS85x5D32_Stride1);
        }
        RUN_KERNEL(DepthwiseConvS85x5D32);
      } if (filter_width == 3 && filter_height == 3 && pad_width <= 1 && pad_height <= 1 && stride_width == 1 && stride_height == 1) {
        RUN_KERNEL(DepthwiseConvS83x3D32_Stride1);
      } if (filter_width == 3 && filter_height == 3 && pad_width <= 1 && pad_height <= 1) {
        RUN_KERNEL(DepthwiseConvS83x3D32);
      }
      RUN_KERNEL(DepthwiseConvS8D32);
    }

    RUN_KERNEL(DepthwiseConvS8Generic);
  }

  RUN_KERNEL(tflite::reference_integer_ops::DepthwiseConvPerChannel);

#undef RUN_KERNEL
}

}  // namespace kelvin::opt
