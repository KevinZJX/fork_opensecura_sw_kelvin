#include "tests/kelvin_isa/kelvin_test.h"

#define vpadd_v(T, Vd, Vs)          \
  {                                 \
    if (sizeof(T) == 2) {           \
      if (std::is_signed<T>::value) \
        vpadd_h_v(Vd, Vs);          \
      else                          \
        vpadd_h_u_v(Vd, Vs);        \
    }                               \
    if (sizeof(T) == 4) {           \
      if (std::is_signed<T>::value) \
        vpadd_w_v(Vd, Vs);          \
      else                          \
        vpadd_w_u_v(Vd, Vs);        \
    }                               \
  }

#define vpadd_v_m(T, Vd, Vs)        \
  {                                 \
    if (sizeof(T) == 2) {           \
      if (std::is_signed<T>::value) \
        vpadd_h_v_m(Vd, Vs);        \
      else                          \
        vpadd_h_u_v_m(Vd, Vs);      \
    }                               \
    if (sizeof(T) == 4) {           \
      if (std::is_signed<T>::value) \
        vpadd_w_v_m(Vd, Vs);        \
      else                          \
        vpadd_w_u_v_m(Vd, Vs);      \
    }                               \
  }

#define vpadd_vv(T, Vd, Vs, Vt)     \
  {                                 \
    if (sizeof(T) == 2) {           \
      if (std::is_signed<T>::value) \
        vpadd_h_vv(Vd, Vs, Vt);     \
      else                          \
        vpadd_h_u_vv(Vd, Vs, Vt);   \
    }                               \
    if (sizeof(T) == 4) {           \
      if (std::is_signed<T>::value) \
        vpadd_w_vv(Vd, Vs, Vt);     \
      else                          \
        vpadd_w_u_vv(Vd, Vs, Vt);   \
    }                               \
  }

#define vpadd_vv_m(T, Vd, Vs, Vt)   \
  {                                 \
    if (sizeof(T) == 2) {           \
      if (std::is_signed<T>::value) \
        vpadd_h_vv_m(Vd, Vs, Vt);   \
      else                          \
        vpadd_h_u_vv_m(Vd, Vs, Vt); \
    }                               \
    if (sizeof(T) == 4) {           \
      if (std::is_signed<T>::value) \
        vpadd_w_vv_m(Vd, Vs, Vt);   \
      else                          \
        vpadd_w_u_vv_m(Vd, Vs, Vt); \
    }                               \
  }

template <typename T1, typename T2>
static void test_vpadd_v() {
  constexpr int n = sizeof(T1);
  int lanes;
  getmaxvl(T2, lanes);

  T1 inp[lanes * 2] __attribute__((aligned(64)));
  T2 ref[lanes] __attribute__((aligned(64)));
  T2 dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes * 2; ++i) {
    inp[i] = ((i & 1 ? 0xc0 : 0x40) << (8 * (n - 1))) + i - 4;
  }

  for (int i = 0; i < lanes; ++i) {
    ref[i] = T2(inp[2 * i + 0]) + T2(inp[2 * i + 1]);
  }

  vld_x(T1, v16, inp);
  vpadd_v(T2, v0, v16);
  vst_x(T2, v0, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vpadd_v[%d] %d %d\n", i, ref[i], dut[i]);
      printf("  inputs: %d, %d\n", inp[2 * i + 0], inp[2 * i + 1]);
      exit(-1);
    }
  }
}

template <typename T1, typename T2>
static void test_vpadd_v_m() {
  constexpr int n = sizeof(T1);
  int lanes;
  getmaxvl_m(T2, lanes);

  T1 inp[lanes * 2] __attribute__((aligned(64)));
  T2 ref[lanes] __attribute__((aligned(64)));
  T2 dut[lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes * 2; ++i) {
    inp[i] = ((i & 1 ? 0xc0 : 0x40) << (8 * (n - 1))) + i - 4;
  }

  for (int i = 0; i < lanes; ++i) {
    ref[i] = T2(inp[2 * i + 0]) + T2(inp[2 * i + 1]);
  }

  vld_x_m(T1, v16, inp);
  vpadd_v_m(T2, v0, v16);
  vst_x_m(T2, v0, dut);

  for (int i = 0; i < lanes; ++i) {
    if (ref[i] != dut[i]) {
      printf("**error vpadd_v_m[%d] %x %x\n", i, ref[i], dut[i]);
      exit(-1);
    }
  }
}

template <typename T1, typename T2>
static void test_vpadd_vv() {
  constexpr int n = sizeof(T1);
  int lanes;
  getmaxvl(T2, lanes);

  T1 inp[2][lanes * 2] __attribute__((aligned(64)));
  T2 ref[2][lanes] __attribute__((aligned(64)));
  T2 dut[2][lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes * 2; ++i) {
    inp[0][i] = ((i & 1 ? 0xc0 : 0x40) << (8 * (n - 1))) + i - 4;
    inp[1][i] = ((i & 1 ? 0x80 : 0x20) << (8 * (n - 1))) + i;
  }

  for (int i = 0; i < lanes; ++i) {
    ref[0][i] = T2(inp[0][2 * i + 0]) + T2(inp[0][2 * i + 1]);
    ref[1][i] = T2(inp[1][2 * i + 0]) + T2(inp[1][2 * i + 1]);
  }

  vld_x(T1, v16, inp[0]);
  vld_x(T1, v20, inp[1]);
  vpadd_vv(T2, v0, v16, v20);
  vst_x(T2, v0, dut[0]);
  vst_x(T2, v1, dut[1]);

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < lanes; ++i) {
      if (ref[j][i] != dut[j][i]) {
        printf("**error vpadd_vv[%d,%d] %x %x\n", j, i, ref[j][i], dut[j][i]);
        exit(-1);
      }
    }
  }
}

template <typename T1, typename T2>
static void test_vpadd_vv_m() {
  constexpr int n = sizeof(T1);
  int lanes;
  getmaxvl_m(T2, lanes);

  T1 inp[2][lanes * 2] __attribute__((aligned(64)));
  T2 ref[2][lanes] __attribute__((aligned(64)));
  T2 dut[2][lanes] __attribute__((aligned(64)));

  for (int i = 0; i < lanes * 2; ++i) {
    inp[0][i] = ((i & 1 ? 0xc0 : 0x40) << (8 * (n - 1))) + i - 4;
    inp[1][i] = ((i & 1 ? 0x80 : 0x20) << (8 * (n - 1))) + i;
  }

  for (int i = 0; i < lanes; ++i) {
    ref[0][i] = T2(inp[0][2 * i + 0]) + T2(inp[0][2 * i + 1]);
    ref[1][i] = T2(inp[1][2 * i + 0]) + T2(inp[1][2 * i + 1]);
  }

  vld_x_m(T1, v16, inp[0]);
  vld_x_m(T1, v20, inp[1]);
  vpadd_vv_m(T2, v0, v16, v20);
  vst_x_m(T2, v0, dut[0]);
  vst_x_m(T2, v4, dut[1]);

  for (int j = 0; j < 2; ++j) {
    for (int i = 0; i < lanes; ++i) {
      if (ref[j][i] != dut[j][i]) {
        printf("**error vpadd_vv_m[%d,%d] %x %x\n", j, i, ref[j][i], dut[j][i]);
        exit(-1);
      }
    }
  }
}

int main() {
  test_vpadd_v<int8_t, int16_t>();
  test_vpadd_v<int16_t, int32_t>();
  test_vpadd_v<uint8_t, uint16_t>();
  test_vpadd_v<uint16_t, uint32_t>();

  test_vpadd_v_m<int8_t, int16_t>();
  test_vpadd_v_m<int16_t, int32_t>();
  test_vpadd_v_m<uint8_t, uint16_t>();
  test_vpadd_v_m<uint16_t, uint32_t>();

  test_vpadd_vv<int8_t, int16_t>();
  test_vpadd_vv<int16_t, int32_t>();
  test_vpadd_vv<uint8_t, uint16_t>();
  test_vpadd_vv<uint16_t, uint32_t>();

  test_vpadd_vv_m<int8_t, int16_t>();
  test_vpadd_vv_m<int16_t, int32_t>();
  test_vpadd_vv_m<uint8_t, uint16_t>();
  test_vpadd_vv_m<uint16_t, uint32_t>();

  return 0;
}
