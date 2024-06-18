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

// Internal helper function within ConvPerChannelD1OD24_5x5.
__attribute__((always_inline)) inline void
ConvPerChannelD1OD24_5x5_inputshuffle() {
  // IN: v48-v52 (modified)
  // OUT: v0-v7
  // CLOBBER: v22-v23, v32-v33

  // Zips row0123 together into v48-v51.
  vzip_b_vv(v22, v48, v50);  // Also writes v23.
  vzip_b_vv(v32, v49, v51);  // Also writes v33.
  vzip_b_vv(v48, v22, v32);  // Also writes v49.
  vzip_b_vv(v50, v23, v33);  // Also writes v51 but it's unused.

  vsliden_h_1_vv(v33, v52, v52);
  vslidep_w_3_vv(v22, v48, v48);
  vslidep_w_1_vv(v23, v48, v48);
  // Patch 0 is row0123[0:19] with row4[0:4].
  vsliden_w_3_vv(v0, v22, v52);  // RHS not yet rotated.
  // Patch 1 is row0123[8:27] with row4[2:6].
  vsliden_w_3_vv(v1, v23, v33);

  vsliden_h_2_vv(v32, v52, v52);
  vsliden_h_3_vv(v33, v52, v52);
  vsliden_w_1_vv(v22, v48, v49);
  vsliden_w_3_vv(v23, v48, v49);
  // Patch 2 is row0123[16:35] with row4[4:8].
  vsliden_w_3_vv(v2, v22, v32);
  // Patch 3 is row0123[24:43] with row4[6:10].
  vsliden_w_3_vv(v3, v23, v33);

  vsliden_h_3_vv(v33, v32, v32);
  vsliden_h_4_vv(v32, v52, v52);
  vslidep_w_3_vv(v22, v49, v49);
  vslidep_w_1_vv(v23, v49, v49);
  // Patch 4 is row0123[32:51] with row4[8:12].
  vsliden_w_3_vv(v4, v22, v32);
  // Patch 5 is row0123[40:59] with row4[10:14].
  vsliden_w_3_vv(v5, v23, v33);

  vsliden_h_3_vv(v33, v32, v32);
  vsliden_w_3_vv(v32, v52, v52);
  vsliden_w_1_vv(v22, v49, v50);
  vsliden_w_3_vv(v23, v49, v50);
  // Patch 6 is row0123[48:67] with row4[12:16].
  vsliden_w_3_vv(v6, v22, v32);
  // Patch 7 is row0123[56:75] with row4[14:18].
  vsliden_w_3_vv(v7, v23, v33);
}

// Internal helper function within ConvPerChannelD1OD24_5x5.
__attribute__((always_inline)) inline void ConvPerChannelD1OD24_5x5_postproc(
    const int32_t* output_multiplier, const int32_t* output_shift,
    int32_t output_offset, int32_t output_activation_min,
    int32_t output_activation_max, int8_t* out_ptr_col0, int8_t* out_ptr_col4) {
  // IN: acc and params
  // OUT: memory, see out_ptr_*
  // CLOBBER: v22-v23, v32-v33, v48-v55

  // Retrieves results.
  vcget(v48);  // v48-v55 is written.

  // Postprocessing and output.
  vevnodd_w_vv(v22, v48, v52);  // Also writes v23.
  vevnodd_w_vv(v32, v49, v53);  // Also writes v33.
  vdmulh_w_rn_vx(v22, v22, output_multiplier[0]);
  vdmulh_w_rn_vx(v23, v23, output_multiplier[4]);
  vdmulh_w_rn_vx(v32, v32, output_multiplier[2]);
  vdmulh_w_rn_vx(v33, v33, output_multiplier[6]);
  vsha_w_r_vx(v22, v22, -output_shift[0]);
  vsha_w_r_vx(v23, v23, -output_shift[4]);
  vsha_w_r_vx(v32, v32, -output_shift[2]);
  vsha_w_r_vx(v33, v33, -output_shift[6]);
  vzip_w_vv(v22, v22, v23);  // Also writes v23.
  vzip_w_vv(v32, v32, v33);  // Also writes v33.
  vmvp_vv(v48, v22, v32);    // Also writes v49.
  vmvp_vv(v52, v23, v33);    // Also writes v53.

  vevnodd_w_vv(v22, v50, v54);  // Also writes v23.
  vevnodd_w_vv(v32, v51, v55);  // Also writes v33.
  vdmulh_w_rn_vx(v22, v22, output_multiplier[1]);
  vdmulh_w_rn_vx(v23, v23, output_multiplier[5]);
  vdmulh_w_rn_vx(v32, v32, output_multiplier[3]);
  vdmulh_w_rn_vx(v33, v33, output_multiplier[7]);
  vsha_w_r_vx(v22, v22, -output_shift[1]);
  vsha_w_r_vx(v23, v23, -output_shift[5]);
  vsha_w_r_vx(v32, v32, -output_shift[3]);
  vsha_w_r_vx(v33, v33, -output_shift[7]);
  vzip_w_vv(v22, v22, v23);  // Also writes v23.
  vzip_w_vv(v32, v32, v33);  // Also writes v33.
  vmvp_vv(v50, v22, v32);    // Also writes v51.
  vmvp_vv(v54, v23, v33);    // Also writes v55.

  vadd_w_vx_m(v48, v48, output_offset);
  vadd_w_vx_m(v52, v52, output_offset);

  vsraqs_b_vx(v48, v48, 0);
  vsraqs_b_vx(v52, v52, 0);
  vstq_b_s_xx(v48, out_ptr_col0, /*output_depth=*/24);
  vstq_b_s_xx(v52, out_ptr_col4, /*output_depth=*/24);
}

}  // namespace

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
  constexpr int stride_width = 2;
  TFLITE_DCHECK_EQ(params.stride_width, stride_width);
  const int stride_height = params.stride_height;
  constexpr int dilation_width_factor = 1;
  TFLITE_DCHECK_EQ(params.dilation_width_factor, dilation_width_factor);
  const int dilation_height_factor = params.dilation_height_factor;
  const int pad_width = params.padding_values.width;
  const int pad_height = params.padding_values.height;
  const int32_t output_offset = params.output_offset;

  // Set min and max value of the output.
  constexpr int32_t output_activation_min = -128;
  TFLITE_DCHECK_EQ(params.quantized_activation_min, output_activation_min);
  constexpr int32_t output_activation_max = 127;
  TFLITE_DCHECK_EQ(params.quantized_activation_max, output_activation_max);

  // Consistency check.
  TFLITE_DCHECK_LE(output_activation_min, output_activation_max);
  TFLITE_DCHECK_EQ(input_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(filter_shape.DimensionsCount(), 4);
  TFLITE_DCHECK_EQ(output_shape.DimensionsCount(), 4);
  const int batches = tflite::MatchingDim(input_shape, 0, output_shape, 0);
  constexpr int input_depth = 1;
  TFLITE_DCHECK_EQ(input_shape.Dims(3), input_depth);
  constexpr const int output_depth = 24;
  TFLITE_DCHECK_EQ(tflite::MatchingDim(filter_shape, 0, output_shape, 3),
                   output_depth);
  if (bias_data) {
    TFLITE_DCHECK_EQ(bias_shape.FlatSize(), output_depth);
  }

  // Check dimensions of the tensors.
  const int input_height = input_shape.Dims(1);
  const int input_width = input_shape.Dims(2);
  constexpr int filter_height = 5;
  TFLITE_DCHECK_EQ(filter_shape.Dims(1), filter_height);
  constexpr int filter_width = 5;
  TFLITE_DCHECK_EQ(filter_shape.Dims(2), filter_width);
  // Input depth is 1 so filter input depth must be 1.
  TFLITE_DCHECK_EQ(filter_shape.Dims(3), 1);
  const int output_height = output_shape.Dims(1);
  const int output_width = output_shape.Dims(2);

  constexpr int patches_per_iteration = 8;
  constexpr int load_width =
      1 + (patches_per_iteration /*=8*/ - 1) * stride_width /*=2*/ +
      (filter_width /*=5*/ - 1) * dilation_width_factor /*=1*/;

  // This optimized path requires all output pixels must read at least 1
  // input pixel.
  TFLITE_DCHECK_LT(pad_height, 5);
  TFLITE_DCHECK_LT(pad_width, 5);
  TFLITE_DCHECK_LT((output_height - 1) * stride_height - pad_height,
                   input_height);
  TFLITE_DCHECK_LT((output_width - 1) * stride_width /*=2*/ - pad_width,
                   input_width);
  // This optimized path is complex enough and we don't wish to handle a load
  // with padding on both sides.
  TFLITE_DCHECK_GE(input_width * input_depth /*=1*/, load_width);

  // Hopefully this lambda helps the compiler get as much done statically
  // as possible.
  auto make_aconv_cmd = [](int in_off) {
    union {
      vconv_u8_t cmd;
      uint32_t raw;
    } cmd;
    vconv_u8_t conv_cmd = {
        .mode = 0,
        .start = 0,
        .stop = 6,  // We're doing [8,25]x[25,8] which is 7 ticks.
        .sbias1 = in_off,
        .sdata1 = true,
        .sbias2 = 0,
        .sdata2 = true,
    };
    cmd.cmd = conv_cmd;
    return cmd.raw;
  };
  auto batch_start_offset = [](int b, int height, int width, int depth) {
    return b * height * width * depth;
  };
  auto row_start_offset = [](int y, int width, int depth) {
    return y * width * depth;
  };
  const uint32_t aconv_cmd = make_aconv_cmd(input_offset);

  int out_y = 0;
  int in_y_min = -pad_height;
  int in_y_max = in_y_min + (filter_height - 1) * dilation_height_factor;

  // Reg Map:
  // v0-v7  : input patches[0..7]
  // v8-v15 : weight for OD[0..7]
  // v16-v21: bias for OD[0..23], each reg is a int32x4 stored twice
  // v22-v23: temp
  // v24-v31: weight for OD[8..15]
  // v32-v34: temp
  // v35-v39: unused
  // v40-v47: weight for OD[16..23]
  // v48-v55: temp
  // v56-v63: unused

  // Temp reg usage:
  // - input loading
  //   - v22-v23: additional loading buffer.
  //   - v32-v34: additional loading buffer.
  //   - v48-v52: input for use in next step.
  //   - v53: scratch pad.
  // - input handling
  //   - v22-v23: scratch pad.
  //   - v32-v33: scratch pad.
  //   - v48-v52: ready-to-use input, with horizontal padding applied where
  //     needed.
  // - aconv
  //   - v48-v55: prepare and set acc.
  // - output
  //   - v22-v23: scratch pad.
  //   - v32-v33: scratch pad.
  //   - v48-v55: get acc and run postproc in place.

  if (bias_data) {
    int32_t od_param_buffer[48] __attribute__((aligned(32)));
    // Preloads bias.
    for (int i = 0; i < 6; i++) {
      od_param_buffer[(8 * i) + 0] = bias_data[(4 * i) + 0];
      od_param_buffer[(8 * i) + 1] = bias_data[(4 * i) + 2];
      od_param_buffer[(8 * i) + 2] = bias_data[(4 * i) + 1];
      od_param_buffer[(8 * i) + 3] = bias_data[(4 * i) + 3];
      od_param_buffer[(8 * i) + 4] = bias_data[(4 * i) + 0];
      od_param_buffer[(8 * i) + 5] = bias_data[(4 * i) + 2];
      od_param_buffer[(8 * i) + 6] = bias_data[(4 * i) + 1];
      od_param_buffer[(8 * i) + 7] = bias_data[(4 * i) + 3];
    }
    vld_b_x_m(v16, &od_param_buffer[0]);
    vld_b_x(v20, &od_param_buffer[32]);
    vld_b_x(v21, &od_param_buffer[40]);
  } else {
    vdup_b_x_m(v16, 0);
    vdup_b_x(v20, 0);
    vdup_b_x(v21, 0);
  }

  {
    // Prepares weight data for preloading. This arrangement is irregular.
    // Logically, this means the following representation for each OD:
    // A0 B0 C0 D0 A1 B1 C1 D1 ... A4 B4 C4 D4 E0 E1 E2 E3 E4 0 (x7)
    // Every 8 OD must then be zipped together into 8 regs (A0*8 B0*8 ......)
    int8_t filter_regs[output_depth /*=24*/ * 32] __attribute__((aligned(32)));
    ::memset(filter_regs, 0, output_depth /*=24*/ * 32);
    for (int ch = 0; ch < output_depth /*=24*/; ++ch) {
      const int regbank = ch / 8;
      const int ch_tail = ch & 0x7;
      for (int y = 0; y < 4; ++y) {  // Row 4 will be handled separately.
        for (int x = 0; x < 5; ++x) {
          filter_regs[regbank * 256 + x * 32 + ch_tail * 4 + y] =
              filter_data[tflite::Offset(filter_shape, ch, y, x, 0)];
        }
      }
      // Reg 5 in each bank is 8ch x E0-E3.
      for (int x = 0; x < 4; ++x) {
        filter_regs[regbank * 256 + 5 * 32 + ch_tail * 4 + x] =
            filter_data[tflite::Offset(filter_shape, ch, 4, x, 0)];
      }
      // Reg 6 in each bank is 8ch x E4.
      filter_regs[regbank * 256 + 6 * 32 + ch_tail * 4] =
          filter_data[tflite::Offset(filter_shape, ch, 4, 4, 0)];
      // Reg 7 is unused and SBZ.
    }
    vld_b_x_m(v8, &filter_regs[0 * 32]);
    vld_b_x_m(v12, &filter_regs[4 * 32]);
    vld_b_x_m(v24, &filter_regs[8 * 32]);
    vld_b_x_m(v28, &filter_regs[12 * 32]);
    vld_b_x_m(v40, &filter_regs[16 * 32]);
    vld_b_x_m(v44, &filter_regs[20 * 32]);
  }

  for (int batch = 0; batch < batches; ++batch) {
    const int8_t* const p_input = &input_data[batch_start_offset(
        batch, input_height, input_width, input_depth /*=1*/)];
    int8_t* const p_output = &output_data[batch_start_offset(
        batch, output_height, output_width, output_depth /*=24*/)];

    // Top loop.
    for (; in_y_min < 0;
         ++out_y, in_y_min += stride_height, in_y_max += stride_height) {
      int out_x_min = 0;
      int out_x_max = patches_per_iteration /*=8*/ - 1;
      int in_x_min = -pad_width;
      int in_x_max = in_x_min + load_width - 1;
      // Top padding is active so we're not going to read the first row.
      const int8_t* in_ptr_row1 =
          &p_input[row_start_offset(in_y_min + 1 * dilation_height_factor,
                                    input_width /*=5*/, input_depth /*=1*/)];
      const int8_t* in_ptr_row2 =
          &p_input[row_start_offset(in_y_min + 2 * dilation_height_factor,
                                    input_width /*=5*/, input_depth /*=1*/)];
      const int8_t* in_ptr_row3 =
          &p_input[row_start_offset(in_y_min + 3 * dilation_height_factor,
                                    input_width /*=5*/, input_depth /*=1*/)];
      const int8_t* in_ptr_row4 = &p_input[row_start_offset(
          in_y_max, input_width /*=5*/, input_depth /*=1*/)];
      int8_t* out_ptr_col0 = &p_output[row_start_offset(out_y, output_width,
                                                        output_depth /*=24*/)];
      int8_t* out_ptr_col4 = out_ptr_col0 + 4 * output_depth /*=24*/;

      // Top left corner.
      // This could only happen [0:1] times.
      if (pad_width > 0) {
        // Same as in_x_max + 1.
        const int true_load_width = load_width - pad_width;

        // Loads all needed rows.
        switch (-in_y_min) {
          case 1:
            vld_b_l_xx(v23, in_ptr_row1, true_load_width);
            [[fallthrough]];
          case 2:
            vld_b_l_xx(v32, in_ptr_row2, true_load_width);
            [[fallthrough]];
          case 3:
            vld_b_l_xx(v33, in_ptr_row3, true_load_width);
            [[fallthrough]];
          case 4:
            vld_b_l_xx(v34, in_ptr_row4, true_load_width);
            break;
          default:
            __builtin_unreachable();
        }
        // Fills padding rows after loading, to improve parallelism.
        // An extra padding-only row to use in horizontal padding.
        vdup_b_x(v53, -input_offset);
        switch (in_y_min) {
          case -4:
            vmv_v(v33, v53);
            [[fallthrough]];
          case -3:
            vmv_v(v32, v53);
            [[fallthrough]];
          case -2:
            vmv_v(v23, v53);
            [[fallthrough]];
          case -1:
            // The first row is vertical padding, no need to pad horizontally.
            vmv_v(v48, v53);
            break;
          default:
            __builtin_unreachable();
        }

        // Applies left padding as needed.
        // Can't pass pad_width because the vslide* wants imm.
        // Can't use vx encoding because scalar is only accepted on the RHS.
        // v53 is -input_offset broadcasted to all lanes.
        switch (pad_width) {
          case 1:
            vslidep_b_1_vv(v49, v53, v23);
            vslidep_b_1_vv(v50, v53, v32);
            vslidep_b_1_vv(v51, v53, v33);
            vslidep_b_1_vv(v52, v53, v34);
            break;
          case 2:
            vslidep_b_2_vv(v49, v53, v23);
            vslidep_b_2_vv(v50, v53, v32);
            vslidep_b_2_vv(v51, v53, v33);
            vslidep_b_2_vv(v52, v53, v34);
            break;
          case 3:
            vslidep_b_3_vv(v49, v53, v23);
            vslidep_b_3_vv(v50, v53, v32);
            vslidep_b_3_vv(v51, v53, v33);
            vslidep_b_3_vv(v52, v53, v34);
            break;
          case 4:
            vslidep_b_4_vv(v49, v53, v23);
            vslidep_b_4_vv(v50, v53, v32);
            vslidep_b_4_vv(v51, v53, v33);
            vslidep_b_4_vv(v52, v53, v34);
            break;
          default:
            __builtin_unreachable();
        }

        // Rearranges input data into place.
        ConvPerChannelD1OD24_5x5_inputshuffle();

        // Computes OD[0..7]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v16, v17);  // Also writes v49.
        vmvp_vv(v50, v16, v17);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v8);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(output_multiplier, output_shift,
                                          output_offset, output_activation_min,
                                          output_activation_max, out_ptr_col0,
                                          out_ptr_col4);

        // Computes OD[8..15]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v18, v19);  // Also writes v49.
        vmvp_vv(v50, v18, v19);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v24);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 8, output_shift + 8, output_offset,
            output_activation_min, output_activation_max, out_ptr_col0 + 8,
            out_ptr_col4 + 8);

        // Computes OD[16..23]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v20, v21);  // Also writes v49.
        vmvp_vv(v50, v20, v21);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v40);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 16, output_shift + 16, output_offset,
            output_activation_min, output_activation_max, out_ptr_col0 + 16,
            out_ptr_col4 + 16);

        // Proceed to next block.
        out_x_min += patches_per_iteration /*=8*/;
        out_x_max += patches_per_iteration /*=8*/;
        in_x_min += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_x_max += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_ptr_row1 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        in_ptr_row2 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        in_ptr_row3 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        in_ptr_row4 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        out_ptr_col0 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
        out_ptr_col4 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
      }

      // Top edge.
      while (out_x_max < output_width && in_x_max < input_width) {
        // Loads all rows that we're not padding.
        switch (-in_y_min) {
          case 1:
            vld_b_l_xx(v49, in_ptr_row1, load_width);
            [[fallthrough]];
          case 2:
            vld_b_l_xx(v50, in_ptr_row2, load_width);
            [[fallthrough]];
          case 3:
            vld_b_l_xx(v51, in_ptr_row3, load_width);
            [[fallthrough]];
          case 4:
            vld_b_l_xx(v52, in_ptr_row4, load_width);
            break;
          default:
            __builtin_unreachable();
        }
        // Fills padding rows.
        switch (in_y_min) {
          case -4:
            vdup_b_x(v51, -input_offset);
            [[fallthrough]];
          case -3:
            vdup_b_x(v50, -input_offset);
            [[fallthrough]];
          case -2:
            vdup_b_x(v49, -input_offset);
            [[fallthrough]];
          case -1:
            vdup_b_x(v48, -input_offset);
            break;
          default:
            __builtin_unreachable();
        }

        // Rearranges input data into place.
        ConvPerChannelD1OD24_5x5_inputshuffle();

        // Computes OD[0..7]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v16, v17);  // Also writes v49.
        vmvp_vv(v50, v16, v17);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v8);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(output_multiplier, output_shift,
                                          output_offset, output_activation_min,
                                          output_activation_max, out_ptr_col0,
                                          out_ptr_col4);

        // Computes OD[8..15]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v18, v19);  // Also writes v49.
        vmvp_vv(v50, v18, v19);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v24);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 8, output_shift + 8, output_offset,
            output_activation_min, output_activation_max, out_ptr_col0 + 8,
            out_ptr_col4 + 8);

        // Computes OD[16..23]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v20, v21);  // Also writes v49.
        vmvp_vv(v50, v20, v21);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v40);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 16, output_shift + 16, output_offset,
            output_activation_min, output_activation_max, out_ptr_col0 + 16,
            out_ptr_col4 + 16);

        // Proceed to next block.
        out_x_min += patches_per_iteration /*=8*/;
        out_x_max += patches_per_iteration /*=8*/;
        in_x_min += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_x_max += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_ptr_row1 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row2 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row3 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row4 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        out_ptr_col0 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
        out_ptr_col4 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
      }

      // Top right corner.
      while (out_x_min < output_width) {
        const int true_patches =
            std::min(output_width - out_x_min, patches_per_iteration);
        const int true_load_width = std::max(input_width - in_x_min, 0);

        // Prepare the selector vector for right padding.
        {
          int8_t selector[32];
          memset(selector, 1, true_load_width);
          memset(selector + true_load_width, 0, 32 - true_load_width);
          vld_b_x(v53, selector);
        }
        // Loads all needed rows and applies right padding.
        switch (-in_y_min) {
          case 1:
            vld_b_l_xx(v49, in_ptr_row1, true_load_width);
            vsel_b_vx(v49, v53, -input_offset);
            [[fallthrough]];
          case 2:
            vld_b_l_xx(v50, in_ptr_row2, true_load_width);
            vsel_b_vx(v50, v53, -input_offset);
            [[fallthrough]];
          case 3:
            vld_b_l_xx(v51, in_ptr_row3, true_load_width);
            vsel_b_vx(v51, v53, -input_offset);
            [[fallthrough]];
          case 4:
            vld_b_l_xx(v52, in_ptr_row4, true_load_width);
            vsel_b_vx(v52, v53, -input_offset);
            break;
          default:
            __builtin_unreachable();
        }
        // Fills padding rows.
        switch (in_y_min) {
          case -4:
            vdup_b_x(v51, -input_offset);
            [[fallthrough]];
          case -3:
            vdup_b_x(v50, -input_offset);
            [[fallthrough]];
          case -2:
            vdup_b_x(v49, -input_offset);
            [[fallthrough]];
          case -1:
            vdup_b_x(v48, -input_offset);
            break;
          default:
            __builtin_unreachable();
        }

        // Rearranges input data into place.
        ConvPerChannelD1OD24_5x5_inputshuffle();

        // We added enough padding to complete a full block, but VSTQ
        // cannot have a write-limiter attached. To workaround this we
        // store to a large-enough buffer and then copy as needed.
        int8_t vstq_buffer[patches_per_iteration /*=8*/ * output_depth /*=24*/]
            __attribute__((aligned(32)));
        int8_t* temp_out_ptr_col0 = &vstq_buffer[0];
        int8_t* temp_out_ptr_col4 = &vstq_buffer[4 * output_depth /*=24*/];

        // Computes OD[0..7]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v16, v17);  // Also writes v49.
        vmvp_vv(v50, v16, v17);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v8);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(output_multiplier, output_shift,
                                          output_offset, output_activation_min,
                                          output_activation_max,
                                          temp_out_ptr_col0, temp_out_ptr_col4);

        // Computes OD[8..15]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v18, v19);  // Also writes v49.
        vmvp_vv(v50, v18, v19);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v24);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 8, output_shift + 8, output_offset,
            output_activation_min, output_activation_max, temp_out_ptr_col0 + 8,
            temp_out_ptr_col4 + 8);

        // Computes OD[16..23]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v20, v21);  // Also writes v49.
        vmvp_vv(v50, v20, v21);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v40);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 16, output_shift + 16, output_offset,
            output_activation_min, output_activation_max,
            temp_out_ptr_col0 + 16, temp_out_ptr_col4 + 16);

        // Copies useful results back.
        ::memcpy(out_ptr_col0, temp_out_ptr_col0,
                 true_patches * output_depth /*=24*/);

        // Proceed to next block.
        // It is easier to use patches_per_iteration instead of true_patches
        // here because the former one is constexpr. These two values will
        // differ iff we've just finished the last block on the row.
        out_x_min += patches_per_iteration /*=8*/;
        out_x_max += patches_per_iteration /*=8*/;
        in_x_min += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_x_max += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_ptr_row1 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row2 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row3 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row4 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        out_ptr_col0 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
        out_ptr_col4 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
      }
    }

    // Main loop (no vertical padding).
    for (; out_y < output_height && in_y_max < input_height;
         ++out_y, in_y_min += stride_height, in_y_max += stride_height) {
      int out_x_min = 0;
      int out_x_max = patches_per_iteration /*=8*/ - 1;
      int in_x_min = -pad_width;
      int in_x_max = in_x_min + load_width - 1;
      const int8_t* in_ptr_row0 = &p_input[row_start_offset(
          in_y_min, input_width /*=5*/, input_depth /*=1*/)];
      const int8_t* in_ptr_row1 =
          &p_input[row_start_offset(in_y_min + 1 * dilation_height_factor,
                                    input_width /*=5*/, input_depth /*=1*/)];
      const int8_t* in_ptr_row2 =
          &p_input[row_start_offset(in_y_min + 2 * dilation_height_factor,
                                    input_width /*=5*/, input_depth /*=1*/)];
      const int8_t* in_ptr_row3 =
          &p_input[row_start_offset(in_y_min + 3 * dilation_height_factor,
                                    input_width /*=5*/, input_depth /*=1*/)];
      const int8_t* in_ptr_row4 = &p_input[row_start_offset(
          in_y_max, input_width /*=5*/, input_depth /*=1*/)];
      int8_t* out_ptr_col0 = &p_output[row_start_offset(out_y, output_width,
                                                        output_depth /*=24*/)];
      int8_t* out_ptr_col4 = out_ptr_col0 + 4 * output_depth /*=24*/;

      // Left edge.
      // This could only happen [0:1] times.
      if (pad_width > 0) {
        // Same as in_x_max + 1.
        const int true_load_width = load_width - pad_width;

        // Loads all needed rows.
        vld_b_l_xx(v22, in_ptr_row0, true_load_width);
        vld_b_l_xx(v23, in_ptr_row1, true_load_width);
        vld_b_l_xx(v32, in_ptr_row2, true_load_width);
        vld_b_l_xx(v33, in_ptr_row3, true_load_width);
        vld_b_l_xx(v34, in_ptr_row4, true_load_width);

        // Applies left padding as needed.
        vdup_b_x(v53, -input_offset);
        // Can't pass pad_width because the vslide* wants imm.
        // Can't use vx encoding because scalar has to be on the RHS.
        // v48 (padded input row 0) is -input_offset broadcasted.
        switch (pad_width) {
          case 1:
            vslidep_b_1_vv(v48, v53, v22);
            vslidep_b_1_vv(v49, v53, v23);
            vslidep_b_1_vv(v50, v53, v32);
            vslidep_b_1_vv(v51, v53, v33);
            vslidep_b_1_vv(v52, v53, v34);
            break;
          case 2:
            vslidep_b_2_vv(v48, v53, v22);
            vslidep_b_2_vv(v49, v53, v23);
            vslidep_b_2_vv(v50, v53, v32);
            vslidep_b_2_vv(v51, v53, v33);
            vslidep_b_2_vv(v52, v53, v34);
            break;
          case 3:
            vslidep_b_3_vv(v48, v53, v22);
            vslidep_b_3_vv(v49, v53, v23);
            vslidep_b_3_vv(v50, v53, v32);
            vslidep_b_3_vv(v51, v53, v33);
            vslidep_b_3_vv(v52, v53, v34);
            break;
          case 4:
            vslidep_b_4_vv(v48, v53, v22);
            vslidep_b_4_vv(v49, v53, v23);
            vslidep_b_4_vv(v50, v53, v32);
            vslidep_b_4_vv(v51, v53, v33);
            vslidep_b_4_vv(v52, v53, v34);
            break;
          default:
            __builtin_unreachable();
        }

        // Rearranges input data into place.
        ConvPerChannelD1OD24_5x5_inputshuffle();

        // Computes OD[0..7]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v16, v17);  // Also writes v49.
        vmvp_vv(v50, v16, v17);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v8);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(output_multiplier, output_shift,
                                          output_offset, output_activation_min,
                                          output_activation_max, out_ptr_col0,
                                          out_ptr_col4);

        // Computes OD[8..15]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v18, v19);  // Also writes v49.
        vmvp_vv(v50, v18, v19);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v24);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 8, output_shift + 8, output_offset,
            output_activation_min, output_activation_max, out_ptr_col0 + 8,
            out_ptr_col4 + 8);

        // Computes OD[16..23]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v20, v21);  // Also writes v49.
        vmvp_vv(v50, v20, v21);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v40);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 16, output_shift + 16, output_offset,
            output_activation_min, output_activation_max, out_ptr_col0 + 16,
            out_ptr_col4 + 16);

        // Proceed to next block.
        out_x_min += patches_per_iteration /*=8*/;
        out_x_max += patches_per_iteration /*=8*/;
        in_x_min += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_x_max += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_ptr_row0 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        in_ptr_row1 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        in_ptr_row2 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        in_ptr_row3 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        in_ptr_row4 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        out_ptr_col0 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
        out_ptr_col4 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
      }

      // Center.
      while (out_x_max < output_width && in_x_max < input_width) {
        // Loads all needed rows.
        // TODO(davidgao): all these reads are misaligned.
        //   Pad the input image may give further speedup.
        vld_b_l_xx(v48, in_ptr_row0, load_width);
        vld_b_l_xx(v49, in_ptr_row1, load_width);
        vld_b_l_xx(v50, in_ptr_row2, load_width);
        vld_b_l_xx(v51, in_ptr_row3, load_width);
        vld_b_l_xx(v52, in_ptr_row4, load_width);

        // Rearranges input data into place.
        ConvPerChannelD1OD24_5x5_inputshuffle();

        // Computes OD[0..7]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v16, v17);  // Also writes v49.
        vmvp_vv(v50, v16, v17);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v8);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(output_multiplier, output_shift,
                                          output_offset, output_activation_min,
                                          output_activation_max, out_ptr_col0,
                                          out_ptr_col4);

        // Computes OD[8..15]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v18, v19);  // Also writes v49.
        vmvp_vv(v50, v18, v19);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v24);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 8, output_shift + 8, output_offset,
            output_activation_min, output_activation_max, out_ptr_col0 + 8,
            out_ptr_col4 + 8);

        // Computes OD[16..23]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v20, v21);  // Also writes v49.
        vmvp_vv(v50, v20, v21);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v40);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 16, output_shift + 16, output_offset,
            output_activation_min, output_activation_max, out_ptr_col0 + 16,
            out_ptr_col4 + 16);

        // Proceed to next block.
        out_x_min += patches_per_iteration /*=8*/;
        out_x_max += patches_per_iteration /*=8*/;
        in_x_min += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_x_max += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_ptr_row0 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row1 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row2 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row3 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row4 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        out_ptr_col0 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
        out_ptr_col4 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
      }

      // Right edge.
      while (out_x_min < output_width) {
        const int true_patches =
            std::min(output_width - out_x_min, patches_per_iteration);
        const int true_load_width = std::max(input_width - in_x_min, 0);

        // Prepare the selector vector for right padding.
        {
          int8_t selector[32];
          memset(selector, 1, true_load_width);
          memset(selector + true_load_width, 0, 32 - true_load_width);
          vld_b_x(v53, selector);
        }
        // Loads all needed rows and applies right padding.
        // Loads all needed rows.
        vld_b_l_xx(v48, in_ptr_row0, true_load_width);
        vld_b_l_xx(v49, in_ptr_row1, true_load_width);
        vld_b_l_xx(v50, in_ptr_row2, true_load_width);
        vld_b_l_xx(v51, in_ptr_row3, true_load_width);
        vld_b_l_xx(v52, in_ptr_row4, true_load_width);
        vsel_b_vx(v48, v53, -input_offset);
        vsel_b_vx(v49, v53, -input_offset);
        vsel_b_vx(v50, v53, -input_offset);
        vsel_b_vx(v51, v53, -input_offset);
        vsel_b_vx(v52, v53, -input_offset);

        // Rearranges input data into place.
        ConvPerChannelD1OD24_5x5_inputshuffle();

        // We added enough padding to complete a full block, but VSTQ
        // cannot have a write-limiter attached. To workaround this we
        // store to a large-enough buffer and then copy as needed.
        int8_t vstq_buffer[patches_per_iteration /*=8*/ * output_depth /*=24*/]
            __attribute__((aligned(32)));
        int8_t* temp_out_ptr_col0 = &vstq_buffer[0];
        int8_t* temp_out_ptr_col4 = &vstq_buffer[4 * output_depth /*=24*/];

        // Computes OD[0..7]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v16, v17);  // Also writes v49.
        vmvp_vv(v50, v16, v17);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v8);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(output_multiplier, output_shift,
                                          output_offset, output_activation_min,
                                          output_activation_max,
                                          temp_out_ptr_col0, temp_out_ptr_col4);

        // Computes OD[8..15]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v18, v19);  // Also writes v49.
        vmvp_vv(v50, v18, v19);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v24);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 8, output_shift + 8, output_offset,
            output_activation_min, output_activation_max, temp_out_ptr_col0 + 8,
            temp_out_ptr_col4 + 8);

        // Computes OD[16..23]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v20, v21);  // Also writes v49.
        vmvp_vv(v50, v20, v21);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v40);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 16, output_shift + 16, output_offset,
            output_activation_min, output_activation_max,
            temp_out_ptr_col0 + 16, temp_out_ptr_col4 + 16);

        // Copies useful results back.
        // TODO(davidgao): this could use some vector copying.
        ::memcpy(out_ptr_col0, temp_out_ptr_col0,
                 true_patches * output_depth /*=24*/);

        // Proceed to next block.
        // It is easier to use patches_per_iteration instead of true_patches
        // here because the former one is constexpr. These two values will
        // differ iff we've just finished the last block on the row.
        out_x_min += patches_per_iteration /*=8*/;
        out_x_max += patches_per_iteration /*=8*/;
        in_x_min += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_x_max += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_ptr_row0 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row1 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row2 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row3 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row4 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        out_ptr_col0 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
        out_ptr_col4 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
      }
    }

    // Bottom loop.
    for (; out_y < output_height;
         ++out_y, in_y_min += stride_height, in_y_max += stride_height) {
      int out_x_min = 0;
      int out_x_max = patches_per_iteration - 1;
      int in_x_min = -pad_width;
      int in_x_max = in_x_min + load_width - 1;
      // Bottom padding is active so we're not going to read the last row.
      const int8_t* in_ptr_row0 = &p_input[row_start_offset(
          in_y_min, input_width /*=5*/, input_depth /*=1*/)];
      const int8_t* in_ptr_row1 = &p_input[row_start_offset(
          in_y_min + 1, input_width /*=5*/, input_depth /*=1*/)];
      const int8_t* in_ptr_row2 = &p_input[row_start_offset(
          in_y_min + 2, input_width /*=5*/, input_depth /*=1*/)];
      const int8_t* in_ptr_row3 = &p_input[row_start_offset(
          in_y_min + 3, input_width /*=5*/, input_depth /*=1*/)];
      int8_t* out_ptr_col0 = &p_output[row_start_offset(out_y, output_width,
                                                        output_depth /*=24*/)];
      int8_t* out_ptr_col4 = out_ptr_col0 + 4 * output_depth /*=24*/;

      // Bottom left corner.
      // This could only happen [0:1] times.
      if (pad_width > 0) {
        // Same as in_x_max + 1.
        const int true_load_width = load_width - pad_width;

        // Loads all needed rows.
        switch (in_y_max - input_height) {
          case 0:
            vld_b_l_xx(v33, in_ptr_row3, true_load_width);
            [[fallthrough]];
          case 1:
            vld_b_l_xx(v32, in_ptr_row2, true_load_width);
            [[fallthrough]];
          case 2:
            vld_b_l_xx(v23, in_ptr_row1, true_load_width);
            [[fallthrough]];
          case 3:
            vld_b_l_xx(v22, in_ptr_row0, true_load_width);
            break;
          default:
            __builtin_unreachable();
        }
        // Fills padding rows.
        vdup_b_x(v53, -input_offset);
        switch (input_height - in_y_max) {
          case -3:
            vmv_v(v23, v53);
            [[fallthrough]];
          case -2:
            vmv_v(v32, v53);
            [[fallthrough]];
          case -1:
            vmv_v(v33, v53);
            [[fallthrough]];
          case 0:
            // The last row always padding-only.
            vmv_v(v52, v53);
            break;
          default:
            __builtin_unreachable();
        }

        // Applies left padding as needed.
        // Can't pass pad_width because the vslide* wants imm.
        // Can't use vx encoding because scalar has to be on the RHS.
        // v53 is -input_offset broadcasted to all lanes.
        switch (pad_width) {
          case 1:
            vslidep_b_1_vv(v48, v53, v22);
            vslidep_b_1_vv(v49, v53, v23);
            vslidep_b_1_vv(v50, v53, v32);
            vslidep_b_1_vv(v51, v53, v33);
            break;
          case 2:
            vslidep_b_2_vv(v48, v53, v22);
            vslidep_b_2_vv(v49, v53, v23);
            vslidep_b_2_vv(v50, v53, v32);
            vslidep_b_2_vv(v51, v53, v33);
            break;
          case 3:
            vslidep_b_3_vv(v48, v53, v22);
            vslidep_b_3_vv(v49, v53, v23);
            vslidep_b_3_vv(v50, v53, v32);
            vslidep_b_3_vv(v51, v53, v33);
            break;
          case 4:
            vslidep_b_4_vv(v48, v53, v22);
            vslidep_b_4_vv(v49, v53, v23);
            vslidep_b_4_vv(v50, v53, v32);
            vslidep_b_4_vv(v51, v53, v33);
            break;
          default:
            __builtin_unreachable();
        }

        // Rearranges input data into place.
        ConvPerChannelD1OD24_5x5_inputshuffle();

        // Computes OD[0..7]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v16, v17);  // Also writes v49.
        vmvp_vv(v50, v16, v17);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v8);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(output_multiplier, output_shift,
                                          output_offset, output_activation_min,
                                          output_activation_max, out_ptr_col0,
                                          out_ptr_col4);

        // Computes OD[8..15]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v18, v19);  // Also writes v49.
        vmvp_vv(v50, v18, v19);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v24);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 8, output_shift + 8, output_offset,
            output_activation_min, output_activation_max, out_ptr_col0 + 8,
            out_ptr_col4 + 8);

        // Computes OD[16..23]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v20, v21);  // Also writes v49.
        vmvp_vv(v50, v20, v21);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v40);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 16, output_shift + 16, output_offset,
            output_activation_min, output_activation_max, out_ptr_col0 + 16,
            out_ptr_col4 + 16);

        // Proceed to next block.
        out_x_min += patches_per_iteration /*=8*/;
        out_x_max += patches_per_iteration /*=8*/;
        in_x_min += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_x_max += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_ptr_row0 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        in_ptr_row1 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        in_ptr_row2 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        in_ptr_row3 +=
            (patches_per_iteration /*=8*/ * stride_width /*=2*/ - pad_width) *
            input_depth /*=1*/;
        out_ptr_col0 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
        out_ptr_col4 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
      }

      // Bottom edge.
      while (out_x_max < output_width && in_x_max < input_width) {
        // Loads all needed rows.
        switch (in_y_max - input_height) {
          case 0:
            vld_b_l_xx(v51, in_ptr_row3, load_width);
            [[fallthrough]];
          case 1:
            vld_b_l_xx(v50, in_ptr_row2, load_width);
            [[fallthrough]];
          case 2:
            vld_b_l_xx(v49, in_ptr_row1, load_width);
            [[fallthrough]];
          case 3:
            vld_b_l_xx(v48, in_ptr_row0, load_width);
            break;
          default:
            __builtin_unreachable();
        }
        // Fills padding rows.
        switch (input_height - in_y_max) {
          case -3:
            vdup_b_x(v49, -input_offset);
            [[fallthrough]];
          case -2:
            vdup_b_x(v50, -input_offset);
            [[fallthrough]];
          case -1:
            vdup_b_x(v51, -input_offset);
            [[fallthrough]];
          case 0:
            vdup_b_x(v52, -input_offset);
            break;
          default:
            __builtin_unreachable();
        }

        // Rearranges input data into place.
        ConvPerChannelD1OD24_5x5_inputshuffle();

        // Computes OD[0..7]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v16, v17);  // Also writes v49.
        vmvp_vv(v50, v16, v17);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v8);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(output_multiplier, output_shift,
                                          output_offset, output_activation_min,
                                          output_activation_max, out_ptr_col0,
                                          out_ptr_col4);

        // Computes OD[8..15]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v18, v19);  // Also writes v49.
        vmvp_vv(v50, v18, v19);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v24);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 8, output_shift + 8, output_offset,
            output_activation_min, output_activation_max, out_ptr_col0 + 8,
            out_ptr_col4 + 8);

        // Computes OD[16..23]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v20, v21);  // Also writes v49.
        vmvp_vv(v50, v20, v21);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v40);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 16, output_shift + 16, output_offset,
            output_activation_min, output_activation_max, out_ptr_col0 + 16,
            out_ptr_col4 + 16);

        // Proceed to next block.
        out_x_min += patches_per_iteration /*=8*/;
        out_x_max += patches_per_iteration /*=8*/;
        in_x_min += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_x_max += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_ptr_row0 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row1 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row2 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row3 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        out_ptr_col0 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
        out_ptr_col4 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
      }

      // Bottom right corner.
      while (out_x_min < output_width) {
        const int true_patches =
            std::min(output_width - out_x_min, patches_per_iteration);
        const int true_load_width = std::max(input_width - in_x_min, 0);

        // Prepare the selector vector for right padding.
        {
          int8_t selector[32];
          memset(selector, 1, true_load_width);
          memset(selector + true_load_width, 0, 32 - true_load_width);
          vld_b_x(v53, selector);
        }
        // Loads all needed rows and applies right padding.
        switch (in_y_max - input_height) {
          case 0:
            vld_b_l_xx(v51, in_ptr_row3, true_load_width);
            vsel_b_vx(v51, v53, -input_offset);
            [[fallthrough]];
          case 1:
            vld_b_l_xx(v50, in_ptr_row2, true_load_width);
            vsel_b_vx(v50, v53, -input_offset);
            [[fallthrough]];
          case 2:
            vld_b_l_xx(v49, in_ptr_row1, true_load_width);
            vsel_b_vx(v49, v53, -input_offset);
            [[fallthrough]];
          case 3:
            vld_b_l_xx(v48, in_ptr_row0, true_load_width);
            vsel_b_vx(v48, v53, -input_offset);
            break;
          default:
            __builtin_unreachable();
        }
        // Fills padding rows.
        switch (input_height - in_y_max) {
          case -3:
            vdup_b_x(v49, -input_offset);
            [[fallthrough]];
          case -2:
            vdup_b_x(v50, -input_offset);
            [[fallthrough]];
          case -1:
            vdup_b_x(v51, -input_offset);
            [[fallthrough]];
          case 0:
            vdup_b_x(v52, -input_offset);
            break;
          default:
            __builtin_unreachable();
        }

        // Rearranges input data into place.
        ConvPerChannelD1OD24_5x5_inputshuffle();

        // We added enough padding to complete a full block, but VSTQ
        // cannot have a write-limiter attached. To workaround this we
        // store to a large-enough buffer and then copy as needed.
        int8_t vstq_buffer[patches_per_iteration /*=8*/ * output_depth /*=24*/]
            __attribute__((aligned(32)));
        int8_t* temp_out_ptr_col0 = &vstq_buffer[0];
        int8_t* temp_out_ptr_col4 = &vstq_buffer[4 * output_depth /*=24*/];

        // Computes OD[0..7]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v16, v17);  // Also writes v49.
        vmvp_vv(v50, v16, v17);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v8);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(output_multiplier, output_shift,
                                          output_offset, output_activation_min,
                                          output_activation_max,
                                          temp_out_ptr_col0, temp_out_ptr_col4);

        // Computes OD[8..15]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v18, v19);  // Also writes v49.
        vmvp_vv(v50, v18, v19);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v24);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 8, output_shift + 8, output_offset,
            output_activation_min, output_activation_max, temp_out_ptr_col0 + 8,
            temp_out_ptr_col4 + 8);

        // Computes OD[16..23]
        // Initializes the accumulators from bias.
        vmvp_vv(v48, v20, v21);  // Also writes v49.
        vmvp_vv(v50, v20, v21);  // Also writes v51.
        vmv_v_m(v52, v48);
        actr_v(v48, v48);  // v48 is read but not written.
        // Performs matmul.
        aconv_vxv(v48, v0, aconv_cmd, v40);  // v48 is not actually written.
        ConvPerChannelD1OD24_5x5_postproc(
            output_multiplier + 16, output_shift + 16, output_offset,
            output_activation_min, output_activation_max,
            temp_out_ptr_col0 + 16, temp_out_ptr_col4 + 16);

        // Copies useful results back.
        ::memcpy(out_ptr_col0, temp_out_ptr_col0,
                 true_patches * output_depth /*=24*/);

        // Proceed to next block.
        // It is easier to use patches_per_iteration instead of true_patches
        // here because the former one is constexpr. These two values will
        // differ iff we've just finished the last block on the row.
        out_x_min += patches_per_iteration /*=8*/;
        out_x_max += patches_per_iteration /*=8*/;
        in_x_min += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_x_max += patches_per_iteration /*=8*/ * stride_width /*=2*/;
        in_ptr_row0 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row1 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row2 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        in_ptr_row3 += (patches_per_iteration /*=8*/ * stride_width /*=2*/) *
                       input_depth /*=1*/;
        out_ptr_col0 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
        out_ptr_col4 += patches_per_iteration /*=8*/ * output_depth /*=24*/;
      }
    }
  }
}

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
