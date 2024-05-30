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

#include "crt/kelvin.h"
#include "tensorflow/lite/kernels/internal/common.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tflm/opt/util.h"

namespace kelvin::opt {

void LogisticS8(int32_t input_zero_point, int32_t input_range_radius,
                int32_t input_multiplier, int32_t input_left_shift,
                int32_t input_size, const int8_t* input_data,
                int8_t* output_data) {
  static constexpr int8_t kMinInt8 = std::numeric_limits<int8_t>::min();
  static constexpr int8_t kMaxInt8 = std::numeric_limits<int8_t>::max();
  static constexpr int32_t kOutputZeroPoint = -128;

#define INPUTS v0
#define MASK_IF_POSITIVE v4
#define MASK_IF_ZERO v8
#define NEG_ABS_INPUT v12
  int i = 0;
  do {
    int count_this_iter = std::min(32L, input_size - i);

    // Load inputs, widen to 32-bits and apply zero-point.
    vld_b_l_xx(INPUTS, input_data + i, count_this_iter);
    vaddw_h_vx(INPUTS, INPUTS, 0);
    vaddw_w_vx(v2, v1, -input_zero_point);
    vaddw_w_vx(INPUTS, v0, -input_zero_point);

    // MultiplyByQuantizedMultiplier
    vmul_w_vx_m(v16, INPUTS, (1 << LEFT_SHIFT(input_left_shift)));
    vdmulh_w_r_vx_m(v16, v16, input_multiplier);
    vsha_w_r_vx_m(v16, v16, RIGHT_SHIFT(input_left_shift));

    // Start gemmlowp::logistic
    // Compute a mask of positive inputs
    constexpr int32_t kInt32_AllOnes = ~0L;
    vgt_w_vx_m(v20, v16, 0);
    vdup_w_x_m(v4, kInt32_AllOnes);
    vsel_w_vx_m(MASK_IF_POSITIVE, v20, 0);

    // Compute a mask of zero inputs
    veq_w_vx_m(v20, v16, 0);
    vdup_w_x_m(v8, -1);
    vsel_w_vx_m(MASK_IF_ZERO, v20, 0);

    // Calculate absolute values of inputs, and negate
    vabsd_w_vx_m(NEG_ABS_INPUT, v16, 0);
    vrsub_w_vx_m(NEG_ABS_INPUT, NEG_ABS_INPUT, 0);

    // Start gemmlowp::exp_on_negative_values
    constexpr int32_t kQ4_OneQuarter = 0x02000000;
    constexpr int32_t kQ4_OneQuarterMinusOne = 0x01FFFFFF;
    vand_w_vx_m(v56, NEG_ABS_INPUT, kQ4_OneQuarterMinusOne);
    vsub_w_vx_m(v56, v56, kQ4_OneQuarter);

    // remainders -- live until after barrel shifters
    vsub_w_vv_m(v60, v56, NEG_ABS_INPUT);

    // Start gemmlowp::exp_on_interval_between_negative_one_quarter_and_0_excl
    vsha_w_r_vx_m(v56, v56, -4);
    constexpr int32_t kQ4_OneEighth = 0x10000000;
    vdup_w_x_m(v20, kQ4_OneEighth);
    vadd_w_vv_m(v56, v56, v20);

    vdmulh_w_r_vv_m(v16, v56, v56);  // x2
    vdmulh_w_r_vv_m(v24, v56, v16);
    vdmulh_w_r_vv_m(v20, v16, v16);
    vsha_w_r_vx_m(v20, v20, 2);

    constexpr int32_t kQ4_ConstantTerm = 0x70f5a894;
    constexpr int32_t kQ4_ConstantOneOverThree = 0x2aaaaaab;
    vadd_w_vv_m(v20, v20, v24);  // x4_over_4 + x3
    vdmulh_w_r_vx_m(v20, v20,
                    kQ4_ConstantOneOverThree);  // _ * constant_1_over_3
    vadd_w_vv_m(v20, v20, v16);                 // _ + x2
    vsha_w_r_vx_m(v20, v20, 1);  // SaturatingRoundingMultiplyByPOT<-1>(_)

    vadd_w_vv_m(v20, v56, v20);                   // x + x4_over_24...
    vdmulh_w_r_vx_m(v20, v20, kQ4_ConstantTerm);  // constant_term * _
    vadd_w_vx_m(v20, v20, kQ4_ConstantTerm);
    // End gemmlowp::exp_on_interval_between_negative_one_quarter_and_0_excl

#define BARREL_SHIFTER(shift, multiplier)  \
  {                                        \
    vand_w_vx_m(v28, v60, 1 << shift);     \
    vne_w_vx_m(v28, v28, 0);               \
    vdmulh_w_r_vx_m(v24, v20, multiplier); \
    vsel_w_vv_m(v24, v28, v20);            \
    vmv_v_m(v20, v24);                     \
  }

    BARREL_SHIFTER(25, 0x63afbe7b);
    BARREL_SHIFTER(26, 0x4da2cbf2);
    BARREL_SHIFTER(27, 0x2f16ac6c);
    BARREL_SHIFTER(28, 0x1152aaa4);
    BARREL_SHIFTER(29, 0x02582ab7);
    BARREL_SHIFTER(30, 0x000afe11);
    BARREL_SHIFTER(0, 0x000000f2);
#undef BARREL_SHIFTER

    constexpr int32_t kResultF_One = 0x7fffffff;
    vne_w_vx_m(v56, NEG_ABS_INPUT, 0);
    vsel_w_vx_m(v24, v56, kResultF_One);
    // End gemmlowp::exp_on_negative_values

    // Begin gemmlowp::one_over_one_plus_x_for_x_in_0_1
    constexpr int32_t kF2_Constant48Over17 = 0x5a5a5a5a;
    constexpr int32_t kF2_ConstantNeg32Over17 = 0xc3c3c3c4;
    constexpr int32_t kF0_OneHalf = 0x40000000;
    vshl_w_vx_m(v24, v24, 1);            // x0 >> 1
    vadd_w_vx_m(v24, v24, kF0_OneHalf);  // _ + ((x1 + 1) >> 1)
    vmv_v_m(v20, v24);                   // half_denominators

    vdmulh_w_r_vx_m(v24, v24,
                    kF2_ConstantNeg32Over17);     // half_denominator * -32/17
    vadd_w_vx_m(v24, v24, kF2_Constant48Over17);  // _ + 48/17

    constexpr int32_t kF2_One = 0x20000000;

#define DIVISION()                  \
  {                                 \
    vdmulh_w_r_vv_m(v28, v24, v20); \
    vmv_v_m(v36, v28);              \
    vgt_w_vx_m(v32, v28, kF2_One);  \
    vsel_w_vx_m(v28, v32, kF2_One); \
    vdup_w_x_m(v32, kF2_One);       \
    vsub_w_vv_m(v40, v28, v32);     \
    vdup_w_x_m(v32, 0xffffffff);    \
    vsub_w_vv_m(v40, v32, v40);     \
    vadd_w_vx_m(v40, v40, 1);       \
    vle_w_vx_m(v32, v36, kF2_One);  \
    vsel_w_vx_m(v36, v32, kF2_One); \
    vdup_w_x_m(v32, kF2_One);       \
    vsub_w_vv_m(v36, v32, v36);     \
    vor_vv_m(v40, v36, v40);        \
    vdmulh_w_r_vv_m(v40, v40, v24); \
    vsha_w_r_vx_m(v40, v40, -2);    \
    vadd_w_vv_m(v40, v40, v24);     \
    vmv_v_m(v24, v40);              \
  }

    DIVISION();
    DIVISION();
    DIVISION();
#undef DIVISION

    vsll_w_vx_m(v40, v40, 1);  // result_if_positive
    // End gemmlowp::one_over_one_plus_x_for_x_in_0_1

    vgt_w_vx_m(v32, v40, kResultF_One);
    vsel_w_vx_m(v44, v32, kResultF_One);  // values >1
    vdup_w_x_m(v32, kResultF_One);
    vsub_w_vv_m(v44, v32, v44);
    vdup_w_x_m(v32, 0xffffffff);
    vsub_w_vv_m(v44, v32, v44);
    vadd_w_vx_m(v44, v44, 1);
    vle_w_vx_m(v36, v40, kResultF_One);
    vsel_w_vx_m(v32, v36, kResultF_One);
    vdup_w_x_m(v36, kResultF_One);
    vsub_w_vv_m(v32, v36, v40);
    vor_vv_m(v44, v44, v32);  // result_if_negative

    vsel_w_vv_m(v40, MASK_IF_POSITIVE, v44);
    vmv_v_m(v56, v40);

    constexpr int32_t kResultF_OneHalf = 0x40000000;
    vdup_w_x_m(v48, kResultF_OneHalf);  // 1/2
    vsel_w_vv_m(v48, MASK_IF_ZERO, v56);
    vmv_v_m(v16, v48);
    // End gemmlowp::logistic

    vle_w_vx_m(v48, INPUTS, -input_range_radius);
    vge_w_vx_m(v56, INPUTS, input_range_radius);
    vor_vv_m(v12, v48, v56);
    vne_w_vx_m(v12, v12, 1);

    vmul_w_vx_m(v48, v48, static_cast<int32_t>(kMinInt8));
    vmul_w_vx_m(v8, v56, static_cast<int32_t>(kMaxInt8));

    vmul_w_vv_m(v16, v16, v12);
    vsha_w_r_vx_m(v16, v16, 23);
    vadd_w_vx_m(v16, v16, kOutputZeroPoint);
    vmax_w_vx_m(v16, v16, kMinInt8);
    vmin_w_vx_m(v16, v16, kMaxInt8);
    vmul_w_vv_m(v12, v12, v16);
    vor_vv_m(v48, v48, v56);
    vor_vv_m(v48, v48, v8);
    vor_vv_m(v48, v48, v12);
    vsraqs_b_vx(v48, v48, 0);
    vst_b_l_xx(v48, output_data + i, count_this_iter);

    i += count_this_iter;
  } while (i < input_size);
}
}  // namespace kelvin::opt
