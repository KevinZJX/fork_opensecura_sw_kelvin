// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Kelvin ISA test common header

#ifndef TESTS_KELVIN_ISA_KELVIN_TEST_H_
#define TESTS_KELVIN_ISA_KELVIN_TEST_H_

#include "crt/kelvin.h"

// Maximum storage required for parameterized machine load/store.
constexpr int VLEN = 256;  // simd register bits. Need to match the HW parameter
constexpr int VLENB = VLEN / 8;
constexpr int VLENH = VLEN / 16;
constexpr int VLENW = VLEN / 32;

#define getmaxvl(T, d)    { if (sizeof(T) == 1) getmaxvl_b(d); if (sizeof(T) == 2) getmaxvl_h(d); if (sizeof(T) == 4) getmaxvl_w(d); }

#define getmaxvl_m(T, d)  { if (sizeof(T) == 1) getmaxvl_b_m(d); if (sizeof(T) == 2) getmaxvl_h_m(d); if (sizeof(T) == 4) getmaxvl_w_m(d); }

#define vld_x(T, Vd, s)   { if (sizeof(T) == 1) vld_b_x(Vd, s); if (sizeof(T) == 2) vld_h_x(Vd, s); if (sizeof(T) == 4) vld_w_x(Vd, s); }

#define vst_x(T, Vd, s)   { if (sizeof(T) == 1) vst_b_x(Vd, s); if (sizeof(T) == 2) vst_h_x(Vd, s); if (sizeof(T) == 4) vst_w_x(Vd, s); }

#define vld_x_m(T, Vd, s)   { if (sizeof(T) == 1) vld_b_x_m(Vd, s); if (sizeof(T) == 2) vld_h_x_m(Vd, s); if (sizeof(T) == 4) vld_w_x_m(Vd, s); }

#define vst_x_m(T, Vd, s)   { if (sizeof(T) == 1) vst_b_x_m(Vd, s); if (sizeof(T) == 2) vst_h_x_m(Vd, s); if (sizeof(T) == 4) vst_w_x_m(Vd, s); }

#define vld_l_xx(T, Vd, s, t)   { if (sizeof(T) == 1) vld_b_l_xx(Vd, s, t); if (sizeof(T) == 2) vld_h_l_xx(Vd, s, t); if (sizeof(T) == 4) vld_w_l_xx(Vd, s, t); }

#define vst_l_xx(T, Vd, s, t)   { if (sizeof(T) == 1) vst_b_l_xx(Vd, s, t); if (sizeof(T) == 2) vst_h_l_xx(Vd, s, t); if (sizeof(T) == 4) vst_w_l_xx(Vd, s, t); }

#define vld_l_xx_m(T, Vd, s, t)   { if (sizeof(T) == 1) vld_b_l_xx_m(Vd, s, t); if (sizeof(T) == 2) vld_h_l_xx_m(Vd, s, t); if (sizeof(T) == 4) vld_w_l_xx_m(Vd, s, t); }

#define vst_l_xx_m(T, Vd, s, t)   { if (sizeof(T) == 1) vst_b_l_xx_m(Vd, s, t); if (sizeof(T) == 2) vst_h_l_xx_m(Vd, s, t); if (sizeof(T) == 4) vst_w_l_xx_m(Vd, s, t); }

#define vld_s_xx(T, Vd, s, t)   { if (sizeof(T) == 1) vld_b_s_xx(Vd, s, t); if (sizeof(T) == 2) vld_h_s_xx(Vd, s, t); if (sizeof(T) == 4) vld_w_s_xx(Vd, s, t); }

#define vst_s_xx(T, Vd, s, t)   { if (sizeof(T) == 1) vst_b_s_xx(Vd, s, t); if (sizeof(T) == 2) vst_h_s_xx(Vd, s, t); if (sizeof(T) == 4) vst_w_s_xx(Vd, s, t); }

#define vld_s_xx_m(T, Vd, s, t)   { if (sizeof(T) == 1) vld_b_s_xx_m(Vd, s, t); if (sizeof(T) == 2) vld_h_s_xx_m(Vd, s, t); if (sizeof(T) == 4) vld_w_s_xx_m(Vd, s, t); }

#define vst_s_xx_m(T, Vd, s, t)   { if (sizeof(T) == 1) vst_b_s_xx_m(Vd, s, t); if (sizeof(T) == 2) vst_h_s_xx_m(Vd, s, t); if (sizeof(T) == 4) vst_w_s_xx_m(Vd, s, t); }

#define vdup_x(T, Vd, s)   { if (sizeof(T) == 1) vdup_b_x(Vd, s); if (sizeof(T) == 2) vdup_h_x(Vd, s); if (sizeof(T) == 4) vdup_w_x(Vd, s); }

#define vdup_x_m(T, Vd, s)   { if (sizeof(T) == 1) vdup_b_x_m(Vd, s); if (sizeof(T) == 2) vdup_h_x_m(Vd, s); if (sizeof(T) == 4) vdup_w_x_m(Vd, s); }

#endif  // TESTS_KELVIN_ISA_KELVIN_TEST_H_
