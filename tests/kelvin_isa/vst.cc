/*
 * Copyright 2023 Google LLC
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

#include "crt/printf_traits.h"
#include "tests/kelvin_isa/kelvin_test.h"

inline uint8_t* vstp_data(uint8_t* data) {
  vst_b_p_x(v0, data);
  return data;
}

inline uint16_t* vstp_data(uint16_t* data) {
  vst_h_p_x(v0, data);
  return data;
}

inline uint32_t* vstp_data(uint32_t* data) {
  vst_w_p_x(v0, data);
  return data;
}

inline uint8_t* vstpm_data(uint8_t* data) {
  vst_b_p_x_m(v0, data);
  return data;
}

inline uint16_t* vstpm_data(uint16_t* data) {
  vst_h_p_x_m(v0, data);
  return data;
}

inline uint32_t* vstpm_data(uint32_t* data) {
  vst_w_p_x_m(v0, data);
  return data;
}

inline uint8_t* vstlp_xx(uint8_t* data, int length) {
  vst_b_lp_xx(v0, data, length);
  return data;
}

inline uint16_t* vstlp_xx(uint16_t* data, int length) {
  vst_h_lp_xx(v0, data, length);
  return data;
}

inline uint32_t* vstlp_xx(uint32_t* data, int length) {
  vst_w_lp_xx(v0, data, length);
  return data;
}

inline uint8_t* vstlpm_xx(uint8_t* data, int length) {
  vst_b_lp_xx_m(v0, data, length);
  return data;
}

inline uint16_t* vstlpm_xx(uint16_t* data, int length) {
  vst_h_lp_xx_m(v0, data, length);
  return data;
}

inline uint32_t* vstlpm_xx(uint32_t* data, int length) {
  vst_w_lp_xx_m(v0, data, length);
  return data;
}

uint8_t* vstp_xx(uint8_t* data, int length) {
  vst_b_p_xx(v0, data, length);
  return data;
}

uint16_t* vstp_xx(uint16_t* data, int length) {
  vst_h_p_xx(v0, data, length);
  return data;
}

uint32_t* vstp_xx(uint32_t* data, int length) {
  vst_w_p_xx(v0, data, length);
  return data;
}

uint8_t* vstpm_xx(uint8_t* data, int length) {
  vst_b_p_xx_m(v0, data, length);
  return data;
}

uint16_t* vstpm_xx(uint16_t* data, int length) {
  vst_h_p_xx_m(v0, data, length);
  return data;
}

uint32_t* vstpm_xx(uint32_t* data, int length) {
  vst_w_p_xx_m(v0, data, length);
  return data;
}

uint8_t* vstsp_xx(uint8_t* data, int stride) {
  vst_b_sp_xx(v0, data, stride);
  return data;
}

uint16_t* vstsp_xx(uint16_t* data, int stride) {
  vst_h_sp_xx(v0, data, stride);
  return data;
}

uint32_t* vstsp_xx(uint32_t* data, int stride) {
  vst_w_sp_xx(v0, data, stride);
  return data;
}

uint8_t* vstspm_xx(uint8_t* data, int stride) {
  vst_b_sp_xx_m(v0, data, stride);
  return data;
}

uint16_t* vstspm_xx(uint16_t* data, int stride) {
  vst_h_sp_xx_m(v0, data, stride);
  return data;
}

uint32_t* vstspm_xx(uint32_t* data, int stride) {
  vst_w_sp_xx_m(v0, data, stride);
  return data;
}

uint8_t* vsttp_xx(uint8_t* data, int stride) {
  vst_b_tp_xx(v0, data, stride);
  return data;
}

uint16_t* vsttp_xx(uint16_t* data, int stride) {
  vst_h_tp_xx(v0, data, stride);
  return data;
}

uint32_t* vsttp_xx(uint32_t* data, int stride) {
  vst_w_tp_xx(v0, data, stride);
  return data;
}

uint8_t* vsttpm_xx(uint8_t* data, int stride) {
  vst_b_tp_xx_m(v0, data, stride);
  return data;
}

uint16_t* vsttpm_xx(uint16_t* data, int stride) {
  vst_h_tp_xx_m(v0, data, stride);
  return data;
}

uint32_t* vsttpm_xx(uint32_t* data, int stride) {
  vst_w_tp_xx_m(v0, data, stride);
  return data;
}

template <typename T>
struct TypeString {
  static const char* type_str() { return "<unknown>"; }
};

template <>
struct TypeString<uint8_t> {
  static const char* type_str() { return "<uint8_t>"; }
};

template <>
struct TypeString<uint16_t> {
  static const char* type_str() { return "<uint16_t>"; }
};

template <>
struct TypeString<uint32_t> {
  static const char* type_str() { return "<uint32_t>"; }
};

// Test vst[_m] instructions for correct pointer update and store result.
template <typename T, bool post_increment, bool stripmine>
void test_ldst_x() {
  uint8_t read_data[4 * VLENB] __attribute__((aligned(64)));
  T* input = reinterpret_cast<T*>(read_data);
  constexpr int size = (stripmine ? 4 : 1) * VLENB / sizeof(T);
  for (int i = 0; i < size; i++) {
    input[i] = static_cast<T>(i);
  }

  if (stripmine) {
    vld_x_m(T, v0, input);
  } else {
    vld_x(T, v0, input);
  }

  uint8_t write_data[4 * VLENB] __attribute__((aligned(64)));
  memset(write_data, 0, sizeof(write_data));
  T* dut = reinterpret_cast<T*>(write_data);
  if (post_increment) {
    if (stripmine) {
      dut = vstpm_data(dut);
    } else {
      dut = vstp_data(dut);
    }
  } else {
    vst_x(T, v0, dut);
  }

  // Test post increment
  int offset = 0;
  if (post_increment) {
    offset = (stripmine ? 4 : 1) * VLENB;
  }
  void* target = reinterpret_cast<void*>(write_data + offset);
  if (reinterpret_cast<void*>(dut) != target) {
    printf("Failed test_ldst_x for type %s:\n", TypeString<T>::type_str());
    printf("  expected dut=%p to be %p\n", reinterpret_cast<void*>(dut),
           target);
    exit(1);
  }

  dut = reinterpret_cast<T*>(write_data);
  for (int i = 0; i < size; i++) {
    if (dut[i] != static_cast<T>(i)) {
      printf("Failed test_ldst_x for type %s:\n", TypeString<T>::type_str());
      printf("  expected %d at position %d/%d got ", i, i, size);
      printf(PrintfTraits<T>::kFmt, dut[i]);
      printf("\n");
      exit(1);
    }
  }
}

// Test vst_p[_m] instructions for correct pointer update and store result.
template <typename T, bool stripmine>
void test_ldst_p_xx(int length) {
  uint8_t read_data[4 * VLENB] __attribute__((aligned(64)));
  T* input = reinterpret_cast<T*>(read_data);
  constexpr int size = (stripmine ? 4 : 1) * VLENB / sizeof(T);
  for (int i = 0; i < size; i++) {
    input[i] = static_cast<T>(i);
  }

  if (stripmine) {
    vld_x_m(T, v0, input);
  } else {
    vld_x(T, v0, input);
  }

  uint8_t write_data[4 * VLENB] __attribute__((aligned(64)));
  memset(write_data, 0, sizeof(write_data));
  T* dut = reinterpret_cast<T*>(write_data);
  if (stripmine) {
    dut = vstpm_xx(dut, length);
  } else {
    dut = vstp_xx(dut, length);
  }

  // Test post increment
  void* target = reinterpret_cast<void*>(write_data + length * sizeof(T));
  if (reinterpret_cast<void*>(dut) != target) {
    printf("Failed test_ldst_p_xx for type %s:\n", TypeString<T>::type_str());
    printf("  expected dut=%p to be %p\n", reinterpret_cast<void*>(dut),
           target);
    exit(1);
  }

  dut = reinterpret_cast<T*>(write_data);
  for (int i = 0; i < size; i++) {
    if (dut[i] != static_cast<T>(i)) {
      printf("Failed test_ldst_p_xx for type %s:\n", TypeString<T>::type_str());
      printf("  expected %d at position %d/%d got ", i, i, size);
      printf(PrintfTraits<T>::kFmt, dut[i]);
      printf("\n");
      exit(1);
    }
  }
}

// Test vst_s[p][_m] instructions for correct pointer update and store result.
template <typename T, bool post_increment, bool stripmine>
void test_ldst_s_xx(int stride) {
  uint8_t read_data[4 * VLENB] __attribute__((aligned(64)));
  T* input = reinterpret_cast<T*>(read_data);
  constexpr int size = (stripmine ? 4 : 1) * VLENB / sizeof(T);
  for (int i = 0; i < size; i++) {
    input[i] = static_cast<T>(i);
  }

  if (stripmine) {
    vld_x_m(T, v0, input);
  } else {
    vld_x(T, v0, input);
  }

  uint8_t write_data[9 * VLENB] __attribute__((aligned(64)));
  memset(write_data, 0, sizeof(write_data));
  T* dut = reinterpret_cast<T*>(write_data);
  if (post_increment) {
    if (stripmine) {
      dut = vstspm_xx(dut, stride);
    } else {
      dut = vstsp_xx(dut, stride);
    }
  } else {
    if (stripmine) {
      vst_s_xx_m(T, v0, dut, stride);
    } else {
      vst_s_xx(T, v0, dut, stride);
    }
  }

  // Test post increment
  int offset = 0;
  if (post_increment) {
    offset = (stripmine ? 4 * stride : stride) * sizeof(T);
  }
  void* target = reinterpret_cast<void*>(write_data + offset);
  if (reinterpret_cast<void*>(dut) != target) {
    printf("Failed test_ldst_s_xx for type %s:\n", TypeString<T>::type_str());
    printf("  expected dut=%p to be %p\n", reinterpret_cast<void*>(dut),
           target);
    exit(1);
  }

  dut = reinterpret_cast<T*>(write_data);
  constexpr int num_vectors = stripmine ? 4 : 1;
  int val = 0;
  for (int v = 0; v < num_vectors; ++v) {
    for (size_t i = 0; i < VLENB / sizeof(T); i++) {
      if (dut[i] != static_cast<T>(val)) {
        printf("Failed test_ldst_s_xx for type %s:\n",
               TypeString<T>::type_str());
        printf("  expected %d at position %d/%d got ", val,
               v * VLENB / sizeof(T) + i, size);
        printf(PrintfTraits<T>::kFmt, dut[i]);
        printf("\n");
        exit(1);
      }
      val++;
    }
    dut += stride;
  }
}

// Test vst_l[p][_m] instructions for correct pointer update and store result.
template <typename T, bool post_increment, bool stripmine>
void test_ldst_l_xx(int length) {
  uint8_t read_data[4 * VLENB] __attribute__((aligned(64)));
  T* input = reinterpret_cast<T*>(read_data);
  constexpr int size = (stripmine ? 4 : 1) * VLENB / sizeof(T);
  for (int i = 0; i < size; i++) {
    input[i] = static_cast<T>(i);
  }

  // Saturate length
  length = std::min(size, length);

  if (stripmine) {
    vld_x_m(T, v0, input);
  } else {
    vld_x(T, v0, input);
  }

  uint8_t write_data[4 * VLENB] __attribute__((aligned(64)));
  memset(write_data, 0, sizeof(write_data));
  T* dut = reinterpret_cast<T*>(write_data);
  if (post_increment) {
    if (stripmine) {
      dut = vstlpm_xx(dut, length);
    } else {
      dut = vstlp_xx(dut, length);
    }
  } else {
    if (stripmine) {
      vst_l_xx_m(T, v0, dut, length);
    } else {
      vst_l_xx(T, v0, dut, length);
    }
  }

  // Test post increment
  void* target = reinterpret_cast<void*>(
      write_data + (post_increment ? length * sizeof(T) : 0));
  if (reinterpret_cast<void*>(dut) != target) {
    printf("Failed test_ldst_l_xx for type %s:\n", TypeString<T>::type_str());
    printf("  expected dut=%p to be %p\n", reinterpret_cast<void*>(dut),
           target);
    exit(1);
  }

  dut = reinterpret_cast<T*>(write_data);
  for (int i = 0; i < length; i++) {
    if (dut[i] != static_cast<T>(i)) {
      printf("Failed test_ldst_l_xx for type %s:\n", TypeString<T>::type_str());
      printf("  expected %d at position %d/%d got ", i, i, size);
      printf(PrintfTraits<T>::kFmt, dut[i]);
      printf("\n");
      exit(1);
    }
  }
}

// Test vst_tp[_m] instructions for correct pointer update and store result.
template <typename T, bool stripmine>
void test_ldst_tp_xx(int stride) {
  uint8_t read_data[4 * VLENB] __attribute__((aligned(64)));
  T* input = reinterpret_cast<T*>(read_data);
  constexpr int size = (stripmine ? 4 : 1) * VLENB / sizeof(T);
  for (int i = 0; i < size; i++) {
    input[i] = static_cast<T>(i);
  }

  if (stripmine) {
    vld_x_m(T, v0, input);
  } else {
    vld_x(T, v0, input);
  }

  uint8_t write_data[9 * VLENB] __attribute__((aligned(64)));
  memset(write_data, 0, sizeof(write_data));
  T* dut = reinterpret_cast<T*>(write_data);

  if (stripmine) {
    dut = vsttpm_xx(dut, stride);
  } else {
    dut = vsttp_xx(dut, stride);
  }

  // Test post increment
  void* target = reinterpret_cast<void*>(write_data + VLENB);
  if (reinterpret_cast<void*>(dut) != target) {
    printf("Failed test_ldst_tp_xx for type %s:\n", TypeString<T>::type_str());
    printf("  expected dut=%p to be %p\n", reinterpret_cast<void*>(dut),
           target);
    exit(1);
  }

  dut = reinterpret_cast<T*>(write_data);
  constexpr int num_vectors = stripmine ? 4 : 1;
  int val = 0;
  for (int v = 0; v < num_vectors; ++v) {
    for (size_t i = 0; i < VLENB / sizeof(T); i++) {
      if (dut[i] != static_cast<T>(val)) {
        printf("Failed test_ldst_tp_xx for type %s:\n",
               TypeString<T>::type_str());
        printf("  expected %d at position %d/%d got ", val,
               v * VLENB / sizeof(T) + i, size);
        printf(PrintfTraits<T>::kFmt, dut[i]);
        printf("\n");
        exit(1);
      }
      val++;
    }
    dut += stride;
  }
}

template <typename T, bool sm>
void test_vst_l() {
  constexpr int n = sizeof(T) * 8;
  constexpr int s = sm ? 4 : 1;
  constexpr int size = s * VLEN / n + 1;
  T in[size] __attribute__((aligned(64)));
  T dut[size] __attribute__((aligned(64)));
  int vlen;
  if (sm) {
    getmaxvl_m(T, vlen);
  } else {
    getmaxvl(T, vlen);
  }

  for (int j = 0; j <= vlen + 1; ++j) {
    in[j] = j + 1;
  }

  if (sm) {
    vdup_x_m(T, v4, 0xcc);
    vld_x_m(T, v0, in);
    vst_x_m(T, v4, dut);
    vst_x_m(T, v4, dut + vlen);
  } else {
    vdup_x(T, v1, 0xcc);
    vld_x(T, v0, in);
    vst_x(T, v1, dut);
    vst_x(T, v1, dut + vlen);
  }

  for (int i = 0; i <= vlen; ++i) {
    if (sm) {
      vst_x_m(T, v4, dut);
      vst_l_xx_m(T, v0, dut, i);
    } else {
      vst_x(T, v1, dut);
      vst_l_xx(T, v0, dut, i);
    }
    for (int j = 0; j <= vlen; ++j) {
      T ref = j < i ? j + 1 : 0xcc;
      if (ref != dut[j]) {
        printf("**error vst_l[%d,%d] ", i, j);
        printf(PrintfTraits<T>::kFmtHex, ref);
        printf(" ");
        printf(PrintfTraits<T>::kFmtHex, dut[j]);
        printf("\n");
        exit(-1);
      }
    }
  }
}

template <typename T>
void test_vst_s() {
  constexpr int n = sizeof(T) * 8;
  constexpr int size = 4 * VLEN / n;
  T in[size] __attribute__((aligned(64)));
  T dut[2 * size] __attribute__((aligned(64)));
  int vlen;
  getmaxvl(T, vlen);

  for (int i = 0; i < 2 * size; ++i) {
    dut[i] = T(0xcccccccc);
  }

  for (int i = 0; i < 4 * vlen; ++i) {
    in[i] = i + 1;
  }

  vld_x_m(T, v0, in);
  vst_s_xx_m(T, v0, dut, vlen + 2);

  for (int s = 0; s < 4; ++s) {
    T* d = dut + s * (vlen + 2);
    for (int i = 0; i < vlen + 2; ++i) {
      T ref = i < vlen ? i + 1 + s * vlen : T(0xcccccccc);
      if (ref != d[i]) {
        printf("**error vst_s[%d] ", i);
        printf(PrintfTraits<T>::kFmtHex, ref);
        printf(" ");
        printf(PrintfTraits<T>::kFmtHex, d[i]);
        printf("\n");
        exit(-1);
      }
    }
  }
}

int main() {
  int vlenb, vlenh, vlenw;

  getmaxvl_b(vlenb);
  getmaxvl_h(vlenh);
  getmaxvl_w(vlenw);

  // vst.*.x
  test_ldst_x<uint8_t, false, false>();
  test_ldst_x<uint16_t, false, false>();
  test_ldst_x<uint32_t, false, false>();

  // vst.*.p.x
  test_ldst_x<uint8_t, true, false>();
  test_ldst_x<uint16_t, true, false>();
  test_ldst_x<uint32_t, true, false>();

  // vst.*.p.x.m
  test_ldst_x<uint8_t, true, true>();
  test_ldst_x<uint16_t, true, true>();
  test_ldst_x<uint32_t, true, true>();

  // vst.*.l.xx
  test_ldst_l_xx<uint8_t, false, false>(1);
  test_ldst_l_xx<uint16_t, false, false>(1);
  test_ldst_l_xx<uint32_t, false, false>(1);
  test_ldst_l_xx<uint8_t, false, false>(vlenb / 2);
  test_ldst_l_xx<uint16_t, false, false>(vlenh / 2);
  test_ldst_l_xx<uint32_t, false, false>(vlenw / 2);
  test_ldst_l_xx<uint8_t, false, false>(vlenb - 1);
  test_ldst_l_xx<uint16_t, false, false>(vlenh - 1);
  test_ldst_l_xx<uint32_t, false, false>(vlenw - 1);
  test_ldst_l_xx<uint8_t, false, false>(vlenb);
  test_ldst_l_xx<uint16_t, false, false>(vlenh);
  test_ldst_l_xx<uint32_t, false, false>(vlenw);
  test_ldst_l_xx<uint8_t, false, false>(vlenb + 1);
  test_ldst_l_xx<uint16_t, false, false>(vlenh + 1);
  test_ldst_l_xx<uint32_t, false, false>(vlenw + 1);

  // vst.*.lp.xx
  test_ldst_l_xx<uint8_t, true, false>(1);
  test_ldst_l_xx<uint16_t, true, false>(1);
  test_ldst_l_xx<uint32_t, true, false>(1);
  test_ldst_l_xx<uint8_t, true, false>(vlenb / 2);
  test_ldst_l_xx<uint16_t, true, false>(vlenh / 2);
  test_ldst_l_xx<uint32_t, true, false>(vlenw / 2);
  test_ldst_l_xx<uint8_t, true, false>(vlenb - 1);
  test_ldst_l_xx<uint16_t, true, false>(vlenh - 1);
  test_ldst_l_xx<uint32_t, true, false>(vlenw - 1);
  test_ldst_l_xx<uint8_t, true, false>(vlenb);
  test_ldst_l_xx<uint16_t, true, false>(vlenh);
  test_ldst_l_xx<uint32_t, true, false>(vlenw);
  test_ldst_l_xx<uint8_t, true, false>(vlenb + 1);
  test_ldst_l_xx<uint16_t, true, false>(vlenh + 1);
  test_ldst_l_xx<uint32_t, true, false>(vlenw + 1);

  // vst.*.p.xx
  test_ldst_p_xx<uint8_t, false>(1);
  test_ldst_p_xx<uint16_t, false>(1);
  test_ldst_p_xx<uint32_t, false>(1);
  test_ldst_p_xx<uint8_t, false>(vlenb / 2);
  test_ldst_p_xx<uint16_t, false>(vlenh / 2);
  test_ldst_p_xx<uint32_t, false>(vlenw / 2);
  test_ldst_p_xx<uint8_t, false>(vlenb - 1);
  test_ldst_p_xx<uint16_t, false>(vlenh - 1);
  test_ldst_p_xx<uint32_t, false>(vlenw - 1);
  test_ldst_p_xx<uint8_t, false>(vlenb);
  test_ldst_p_xx<uint16_t, false>(vlenh);
  test_ldst_p_xx<uint32_t, false>(vlenw);
  test_ldst_p_xx<uint8_t, false>(vlenb + 1);
  test_ldst_p_xx<uint16_t, false>(vlenh + 1);
  test_ldst_p_xx<uint32_t, false>(vlenw + 1);

  // vst.*.sp.xx. Stride should be >= vector size.
  test_ldst_s_xx<uint8_t, true, false>(vlenb);
  test_ldst_s_xx<uint16_t, true, false>(vlenh);
  test_ldst_s_xx<uint32_t, true, false>(vlenw);
  test_ldst_s_xx<uint8_t, true, false>(vlenb + 1);
  test_ldst_s_xx<uint16_t, true, false>(vlenh + 1);
  test_ldst_s_xx<uint32_t, true, false>(vlenw + 1);

  // vst.*.tp.xx. Stride should be >= vector size.
  test_ldst_tp_xx<uint8_t, false>(vlenb);
  test_ldst_tp_xx<uint16_t, false>(vlenh);
  test_ldst_tp_xx<uint32_t, false>(vlenw);
  test_ldst_tp_xx<uint8_t, false>(vlenb + 1);
  test_ldst_tp_xx<uint16_t, false>(vlenh + 1);
  test_ldst_tp_xx<uint32_t, false>(vlenw + 1);

  getmaxvl_b_m(vlenb);
  getmaxvl_h_m(vlenh);
  getmaxvl_w_m(vlenw);

  // vst.*.l.xx.m
  test_ldst_l_xx<uint8_t, false, true>(1);
  test_ldst_l_xx<uint16_t, false, true>(1);
  test_ldst_l_xx<uint32_t, false, true>(1);
  test_ldst_l_xx<uint8_t, false, true>(vlenb / 2);
  test_ldst_l_xx<uint16_t, false, true>(vlenh / 2);
  test_ldst_l_xx<uint32_t, false, true>(vlenw / 2);
  test_ldst_l_xx<uint8_t, false, true>(vlenb - 1);
  test_ldst_l_xx<uint16_t, false, true>(vlenh - 1);
  test_ldst_l_xx<uint32_t, false, true>(vlenw - 1);
  test_ldst_l_xx<uint8_t, false, true>(vlenb);
  test_ldst_l_xx<uint16_t, false, true>(vlenh);
  test_ldst_l_xx<uint32_t, false, true>(vlenw);
  test_ldst_l_xx<uint8_t, false, true>(vlenb + 1);
  test_ldst_l_xx<uint16_t, false, true>(vlenh + 1);
  test_ldst_l_xx<uint32_t, false, true>(vlenw + 1);

  // vst.*.lp.xx.m
  test_ldst_l_xx<uint8_t, true, true>(1);
  test_ldst_l_xx<uint16_t, true, true>(1);
  test_ldst_l_xx<uint32_t, true, true>(1);
  test_ldst_l_xx<uint8_t, true, true>(vlenb / 2);
  test_ldst_l_xx<uint16_t, true, true>(vlenh / 2);
  test_ldst_l_xx<uint32_t, true, true>(vlenw / 2);
  test_ldst_l_xx<uint8_t, true, true>(vlenb - 1);
  test_ldst_l_xx<uint16_t, true, true>(vlenh - 1);
  test_ldst_l_xx<uint32_t, true, true>(vlenw - 1);
  test_ldst_l_xx<uint8_t, true, true>(vlenb);
  test_ldst_l_xx<uint16_t, true, true>(vlenh);
  test_ldst_l_xx<uint32_t, true, true>(vlenw);
  test_ldst_l_xx<uint8_t, true, true>(vlenb + 1);
  test_ldst_l_xx<uint16_t, true, true>(vlenh + 1);
  test_ldst_l_xx<uint32_t, true, true>(vlenw + 1);

  // vst.*.p.xx.m
  test_ldst_p_xx<uint8_t, true>(vlenb);
  test_ldst_p_xx<uint16_t, true>(vlenh);
  test_ldst_p_xx<uint32_t, true>(vlenw);
  test_ldst_p_xx<uint8_t, true>(vlenb + 1);
  test_ldst_p_xx<uint16_t, true>(vlenh + 1);
  test_ldst_p_xx<uint32_t, true>(vlenw + 1);

  // vst.*.s.xx.m. Stride should be >= vector size.
  test_ldst_s_xx<uint8_t, false, true>(vlenb);
  test_ldst_s_xx<uint16_t, false, true>(vlenh);
  test_ldst_s_xx<uint32_t, false, true>(vlenw);
  test_ldst_s_xx<uint8_t, false, true>(vlenb + 1);
  test_ldst_s_xx<uint16_t, false, true>(vlenh + 1);
  test_ldst_s_xx<uint32_t, false, true>(vlenw + 1);

  // vst.*.sp.xx.m. Stride should be >= vector size.
  test_ldst_s_xx<uint8_t, true, true>(vlenb);
  test_ldst_s_xx<uint16_t, true, true>(vlenh);
  test_ldst_s_xx<uint32_t, true, true>(vlenw);
  test_ldst_s_xx<uint8_t, true, true>(vlenb + 1);
  test_ldst_s_xx<uint16_t, true, true>(vlenh + 1);
  test_ldst_s_xx<uint32_t, true, true>(vlenw + 1);

  // vst.*.tp.xx.m. Stride should be >= vector size.
  test_ldst_tp_xx<uint8_t, true>(vlenb);
  test_ldst_tp_xx<uint16_t, true>(vlenh);
  test_ldst_tp_xx<uint32_t, true>(vlenw);
  test_ldst_tp_xx<uint8_t, true>(vlenb + 1);
  test_ldst_tp_xx<uint16_t, true>(vlenh + 1);
  test_ldst_tp_xx<uint32_t, true>(vlenw + 1);

  test_vst_l<uint8_t, 0>();
  test_vst_l<uint16_t, 0>();
  test_vst_l<uint32_t, 0>();

  test_vst_l<uint8_t, 1>();
  test_vst_l<uint16_t, 1>();
  test_vst_l<uint32_t, 1>();

  test_vst_s<uint8_t>();
  test_vst_s<uint16_t>();
  test_vst_s<uint32_t>();

  return 0;
}
