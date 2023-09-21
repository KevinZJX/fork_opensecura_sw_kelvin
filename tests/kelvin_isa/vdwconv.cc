// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include <cstdio>
#include <cstring>

#include "tests/kelvin_isa/kelvin_test.h"
#include "tests/kelvin_isa/vdwconv_data.h"

struct vdwconv_u8_t {
  uint32_t mode:2;      // 1:0
  uint32_t sparsity:2;  // 3:2
  uint32_t regbase:4;   // 7:4
  uint32_t rsvd:4;      // 11:8
  int32_t sbias1:9;    // 20:12
  uint32_t sdata1:1;    // 21
  int32_t sbias2:9;    // 30:22
  uint32_t sdata2:1;    // 31
};
static_assert(sizeof(vdwconv_u8_t) == 4);

#ifdef TEST_GEN
static int32_t dwconv(const vdwconv_u8_t& cmd, uint8_t ina[3], uint8_t inb[3]) {
  int32_t sbias1 = cmd.sbias1;
  int32_t sbias2 = cmd.sbias2;
  int32_t accum = 0;
  for (int i = 0; i < 3; ++i) {
    int32_t sdata1 = cmd.sdata1 ? int8_t(ina[i]) : uint8_t(ina[i]);
    int32_t sdata2 = cmd.sdata2 ? int8_t(inb[i]) : uint8_t(inb[i]);
    accum += (sdata1 + sbias1) * (sdata2 + sbias2);
  }
  return accum;
}
#endif  // TEST_GEN

void dwconv(const vdwconv_u8_t& cmd, const uint8_t ina[3][kZlen],
            const uint8_t inb[3][kZlen], const uint32_t ref[4][kZlen / 4],
            uint32_t dut[4][kZlen / 4]) {
  uint32_t cmdw;
  memcpy(&cmdw, &cmd, 4);

  int sparsity = cmd.sparsity;
  int regbase = cmd.regbase;

#ifdef TEST_GEN
  for (int j = 0; j < kZlen / 4; ++j) {
    for (int i = 0; i < 4; ++i) {
      int idx = i + 4 * j;
      uint8_t va[3];
      uint8_t vb[3] = {inb[0][idx], inb[1][idx], inb[2][idx]};
      if (sparsity == 0) {
        va[0] = ina[0][idx];
        va[1] = ina[1][idx];
        va[2] = ina[2][idx];
      } else if (sparsity == 1) {
        va[0] = ina[1][idx - 4];
        va[1] = ina[1][idx + 0];
        va[2] = ina[1][idx + 4];
      } else if (sparsity == 2) {
        va[0] = ina[0][idx + 0];
        va[1] = ina[0][idx + 4];
        va[2] = ina[0][idx + 8];
      }
      const int interleave[4] = {0, 2, 1, 3};
      ref[interleave[i]][j] = dwconv(cmd, va, vb);
    }
  }
#endif  // TEST_GEN

  int vlenb;
  getmaxvl_b(vlenb);

  vdup_b_x(v16, 0);
  vdup_b_x(v17, 0);
  vdup_b_x(v18, 0);
  vdup_b_x(v19, 0);
  vdup_b_x(v20, 0);
  vdup_b_x(v21, 0);
  vdup_b_x(v22, 0);
  vdup_b_x(v23, 0);
  vdup_b_x(v24, 0);

  vdup_w_x_m(v52, 0);

  for (int i = 0; i < kZlen; i += vlenb) {
    const int j = i / 4;

    // dense
    const uint8_t* pp = ina[0] + i;  // prev
    const uint8_t* pc = ina[1] + i;  // curr
    const uint8_t* pn = ina[2] + i;  // next

    if (sparsity == 1) {
      pp = ina[1] + i - vlenb;  // prev
      pc = ina[1] + i;          // curr
      pn = ina[1] + i + vlenb;  // next
    }

    if (sparsity == 2) {
      pp = ina[0] + i;          // curr
      pc = ina[0] + i + vlenb;  // next
      pn = ina[0] + i + vlenb;  // unused
    }

    switch (regbase) {
      case 0:
        vld_b_x(v16, pp);
        vld_b_x(v17, pc);
        vld_b_x(v18, pn);
        break;
      case 1:
        vld_b_x(v17, pp);
        vld_b_x(v18, pc);
        vld_b_x(v19, pn);
        break;
      case 2:
        vld_b_x(v18, pp);
        vld_b_x(v19, pc);
        vld_b_x(v20, pn);
        break;
      case 3:
        vld_b_x(v19, pp);
        vld_b_x(v20, pc);
        vld_b_x(v21, pn);
        break;
      case 4:
        vld_b_x(v20, pp);
        vld_b_x(v21, pc);
        vld_b_x(v22, pn);
        break;
      case 5:
        vld_b_x(v21, pp);
        vld_b_x(v22, pc);
        vld_b_x(v23, pn);
        break;
      case 6:
        vld_b_x(v22, pp);
        vld_b_x(v23, pc);
        vld_b_x(v24, pn);
        break;
      case 7:
        vld_b_x(v17, pp);
        vld_b_x(v16, pc);
        vld_b_x(v18, pn);
        break;
      case 8:
        vld_b_x(v17, pp);
        vld_b_x(v18, pc);
        vld_b_x(v16, pn);
        break;
      case 9:
        vld_b_x(v19, pp);
        vld_b_x(v20, pc);
        vld_b_x(v16, pn);
        break;
      case 10:
        vld_b_x(v21, pp);
        vld_b_x(v22, pc);
        vld_b_x(v16, pn);
        break;
      case 11:
        vld_b_x(v23, pp);
        vld_b_x(v24, pc);
        vld_b_x(v16, pn);
        break;
      case 12:
        vld_b_x(v18, pp);
        vld_b_x(v16, pc);
        vld_b_x(v17, pn);
        break;
      case 13:
        vld_b_x(v20, pp);
        vld_b_x(v16, pc);
        vld_b_x(v17, pn);
        break;
      case 14:
        vld_b_x(v22, pp);
        vld_b_x(v16, pc);
        vld_b_x(v17, pn);
        break;
      case 15:
        vld_b_x(v24, pp);
        vld_b_x(v16, pc);
        vld_b_x(v17, pn);
        break;
      default:
        exit(-1);
        break;
    }

    vld_b_x(v32, inb[0] + i);
    vld_b_x(v33, inb[1] + i);
    vld_b_x(v34, inb[2] + i);

    adwinit_v(v48, v52);
    vdwconv_vxv(v48, v16, cmdw, v32);

    vst_w_x(v48, dut[0] + j);
    vst_w_x(v49, dut[1] + j);
    vst_w_x(v50, dut[2] + j);
    vst_w_x(v51, dut[3] + j);
  }

#ifdef TEST_GEN
  // Print the results.
  printf("{ %d, %d, %d, %d, ", cmd.sdata1, cmd.sdata2, cmd.sbias1, cmd.sbias2);
  printf("{ ");
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < kZlen / 4; ++j) {
      printf("0x%lx, ", ref[i][j]);
    }
  }
  printf("} },\n");
#else
  // Check the results.
  for (int i = 0; i < 4; ++i) {
    for (int j = 0; j < kZlen / 4; ++j) {
      uint32_t r = ref[i][j];
      uint32_t d = dut[i][j];
      if (r != d) {
        printf("**error::test_dwconv(%d,%d)(%d,%d,%d,%d)[%d,%d] ", cmd.sparsity,
               cmd.regbase, cmd.sdata1, cmd.sdata2, cmd.sbias1, cmd.sbias2, i,
               j);
        printf("0x%lx 0x%lx\n", r, d);
        exit(-1);
      }
    }
  }
#endif  // TEST_GEN
}

template <int step, bool use_accum>
void test_vdwconv() {
  const uint32_t ref1[4] = {0x00001ba4, 0x01021ea8, 0x00061bac, 0x00001bb0};
  const uint32_t ref2[4] = {0x00004dbb, 0x010250bf, 0x00064dc3, 0x00004dc7};
  const uint32_t ref3[4] = {0x000066af, 0x010269b3, 0x000666b7, 0x000066bb};
  const uint32_t* ref = step == 1 ? ref1 : step == 2 ? ref2 : ref3;
  uint32_t dut[4];
  uint32_t cmdw = 0;

  vdup_w_x(v12, 0x00000000);
  vdup_w_x(v13, 0x01020304);
  vdup_w_x(v14, 0x00060008);
  vdup_w_x(v15, 0x0000000c);

  vdup_b_x(v16, 23);
  vdup_b_x(v17, 34);
  vdup_b_x(v18, 45);

  vdup_b_x(v32, 56);
  vdup_b_x(v33, 67);
  vdup_b_x(v34, 78);

  vdup_b_x(v36, 76);
  vdup_b_x(v37, 65);
  vdup_b_x(v38, 54);

  adwinit_v(v0, v12);
  if (!use_accum) {
    vdwconv_vxv(v0, v16, cmdw, v32);
    if (step >= 2) {
      vdwconv_vxv(v0, v32, cmdw, v36);
    }
    if (step >= 3) {
      vdwconv_vxv(v0, v36, cmdw, v16);
    }
  } else {
    if (step == 1) {
      vdwconv_vxv(v0, v16, cmdw, v32);
    } else if (step == 2) {
      adwconv_vxv(v0, v16, cmdw, v32);
      vdwconv_vxv(v0, v32, cmdw, v36);
    } else if (step == 3) {
      adwconv_vxv(v0, v16, cmdw, v32);
      adwconv_vxv(v0, v32, cmdw, v36);
      vdwconv_vxv(v0, v36, cmdw, v16);
    }
  }

  vst_w_l_xx(v0, dut + 0, 1);
  vst_w_l_xx(v1, dut + 1, 1);
  vst_w_l_xx(v2, dut + 2, 1);
  vst_w_l_xx(v3, dut + 3, 1);

  for (int i = 0; i < 4; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error::test_dwconv<%d,%d>[%d] 0x%lx 0x%lx\n", step, use_accum,
             i, ref[i], dut[i]);
      exit(-1);
    }
  }
}

void test_vdwconv(int sparsity, int regbase, const test_dwconv_t& test) {
  uint32_t dut[4][kZlen / 4];
  vdwconv_u8_t cmd;

  cmd.mode = 0;
  cmd.sparsity = sparsity;
  cmd.regbase = regbase;
  cmd.sbias1 = test.sbias1;
  cmd.sdata1 = test.sdata1;
  cmd.sbias2 = test.sbias2;
  cmd.sdata2 = test.sdata2;

  dwconv(cmd, ina_, inb_, test.data, dut);
}

int main() {
#ifdef TEST_GEN
  uint32_t* pw_ina = reinterpret_cast<uint32_t*>(ina_);
  uint32_t* pw_inb = reinterpret_cast<uint32_t*>(inb_);
  uint8_t* pb_ina = reinterpret_cast<uint8_t*>(ina_);
  uint8_t* pb_inb = reinterpret_cast<uint8_t*>(inb_);
  for (int i = 0; i < 3 * kZlen / 4; ++i) {
    pw_ina[i] = krand();
    pw_inb[i] = krand();
  }
  printf("{ ");
  for (int i = 0; i < 3 * kZlen; ++i) {
    printf("0x%02x, ", pb_ina[i]);
  }
  printf("};\n");
  printf("{ ");
  for (int i = 0; i < 3 * kZlen; ++i) {
    printf("0x%02x, ", pb_inb[i]);
  }
  printf("};\n");
#endif  // TEST_GEN

  // Accumulator test.
  test_vdwconv<1, false>();
  test_vdwconv<2, false>();
  test_vdwconv<3, false>();

  test_vdwconv<1, true>();
  test_vdwconv<2, true>();
  test_vdwconv<3, true>();

  // Regbase tests.
  for (int i = 0; i < 16; ++i) {
    test_vdwconv(0, i, ref_[0]);
  }

  // Bias tests.
  for (int i = 0; i < 20; ++i) {
    test_vdwconv(0, 0, ref_[i]);
  }

  // Sparsity tests.
  test_vdwconv(1, 0, ref_sparsity1_[0]);
  test_vdwconv(2, 0, ref_sparsity2_[0]);

  return 0;
}
