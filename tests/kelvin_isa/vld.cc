#include "tests/kelvin_isa/kelvin_test.h"

inline uint8_t* vldp_data(uint8_t* data) {
  vld_b_p_x(v0, data);
  return data;
}

inline uint16_t* vldp_data(uint16_t* data) {
  vld_h_p_x(v0, data);
  return data;
}

inline uint32_t* vldp_data(uint32_t* data) {
  vld_w_p_x(v0, data);
  return data;
}

inline uint8_t* vldpm_data(uint8_t* data) {
  vld_b_p_x_m(v0, data);
  return data;
}

inline uint16_t* vldpm_data(uint16_t* data) {
  vld_h_p_x_m(v0, data);
  return data;
}

inline uint32_t* vldpm_data(uint32_t* data) {
  vld_w_p_x_m(v0, data);
  return data;
}

uint8_t* vldp_xx(uint8_t* data, int length) {
  vld_b_p_xx(v0, data, length);
  return data;
}

uint16_t* vldp_xx(uint16_t* data, int length) {
  vld_h_p_xx(v0, data, length);
  return data;
}

uint32_t* vldp_xx(uint32_t* data, int length) {
  vld_w_p_xx(v0, data, length);
  return data;
}

uint8_t* vldpm_xx(uint8_t* data, int length) {
  vld_b_p_xx_m(v0, data, length);
  return data;
}

uint16_t* vldpm_xx(uint16_t* data, int length) {
  vld_h_p_xx_m(v0, data, length);
  return data;
}

uint32_t* vldpm_xx(uint32_t* data, int length) {
  vld_w_p_xx_m(v0, data, length);
  return data;
}

inline uint8_t* vldlp_data(uint8_t* data, int length) {
  vld_b_lp_xx(v0, data, length);
  return data;
}

inline uint16_t* vldlp_data(uint16_t* data, int length) {
  vld_h_lp_xx(v0, data, length);
  return data;
}

inline uint32_t* vldlp_data(uint32_t* data, int length) {
  vld_w_lp_xx(v0, data, length);
  return data;
}

inline uint8_t* vldlpm_data(uint8_t* data, int length) {
  vld_b_lp_xx_m(v0, data, length);
  return data;
}

inline uint16_t* vldlpm_data(uint16_t* data, int length) {
  vld_h_lp_xx_m(v0, data, length);
  return data;
}

inline uint32_t* vldlpm_data(uint32_t* data, int length) {
  vld_w_lp_xx_m(v0, data, length);
  return data;
}

uint8_t* vldsp_xx(uint8_t* data, int stride) {
  vld_b_sp_xx(v0, data, stride);
  return data;
}

uint16_t* vldsp_xx(uint16_t* data, int stride) {
  vld_h_sp_xx(v0, data, stride);
  return data;
}

uint32_t* vldsp_xx(uint32_t* data, int stride) {
  vld_w_sp_xx(v0, data, stride);
  return data;
}

uint8_t* vldspm_xx(uint8_t* data, int stride) {
  vld_b_sp_xx_m(v0, data, stride);
  return data;
}

uint16_t* vldspm_xx(uint16_t* data, int stride) {
  vld_h_sp_xx_m(v0, data, stride);
  return data;
}

uint32_t* vldspm_xx(uint32_t* data, int stride) {
  vld_w_sp_xx_m(v0, data, stride);
  return data;
}

uint8_t* vldtp_xx(uint8_t* data, int stride) {
  vld_b_tp_xx(v0, data, stride);
  return data;
}

uint16_t* vldtp_xx(uint16_t* data, int stride) {
  vld_h_tp_xx(v0, data, stride);
  return data;
}

uint32_t* vldtp_xx(uint32_t* data, int stride) {
  vld_w_tp_xx(v0, data, stride);
  return data;
}

uint8_t* vldtpm_xx(uint8_t* data, int stride) {
  vld_b_tp_xx_m(v0, data, stride);
  return data;
}

uint16_t* vldtpm_xx(uint16_t* data, int stride) {
  vld_h_tp_xx_m(v0, data, stride);
  return data;
}

uint32_t* vldtpm_xx(uint32_t* data, int stride) {
  vld_w_tp_xx_m(v0, data, stride);
  return data;
}

template<typename T>
struct TypeString {
  static const char* type_str() {
    return "<unknown>";
  }
};

template<>
struct TypeString<uint8_t> {
  static const char* type_str() {
    return "<uint8_t>";
  }
};

template<>
struct TypeString<uint16_t> {
  static const char* type_str() {
    return "<uint16_t>";
  }
};

template<>
struct TypeString<uint32_t> {
  static const char* type_str() {
    return "<uint32_t>";
  }
};

// Test vld[_m] instructions for correct pointer update and load result.
template <typename T, bool post_increment, bool stripmine>
void test_ldst_x() {
  uint8_t read_data[4*VLENB] __attribute__((aligned(64)));
  T* dut = reinterpret_cast<T*>(read_data);
  constexpr int size = (stripmine ? 4 : 1) * VLENB / sizeof(T);
  for (int i = 0; i < size; i++) {
    dut[i] = static_cast<T>(i);
  }

  if (post_increment) {
    if (stripmine) {
      dut = vldpm_data(dut);
    } else {
      dut = vldp_data(dut);
    }
  } else {
    vld_x(T, v0, dut);
  }

  // Test post increment
  int offset = 0;
  if (post_increment) {
    offset = (stripmine ? 4 : 1) * VLENB;
  }
  void* target = reinterpret_cast<void*>(read_data + offset);
  if (reinterpret_cast<void*>(dut) != target) {
    printf("Failed test_ldst_x for type %s:\n", TypeString<T>::type_str());
    printf("  expected dut=0x%x to be 0x%x\n",
           reinterpret_cast<void*>(dut), target);
    exit(1);
  }

  uint8_t write_data[4*VLENB] __attribute__((aligned(64)));
  T* result = reinterpret_cast<T*>(write_data);
  if (stripmine) {
    vst_x_m(T, v0, result);
  } else {
    vst_x(T, v0, result);
  }

  for (int i = 0; i < size; i++) {
    if (result[i] != static_cast<T>(i)) {
      printf("Failed test_ldst_x for type %s:\n", TypeString<T>::type_str());
      printf("  expected %d at position %d/%d got %d\n", i, i, size, result[i]);
      exit(1);
    }
  }
}

// Test vld_p[_m] instructions for correct pointer update and load result.
template<typename T, bool stripmine>
void test_ldst_p_xx(int length) {
  uint8_t read_data[4*VLENB] __attribute__((aligned(64)));
  T* dut = reinterpret_cast<T*>(read_data);
  constexpr int size = (stripmine ? 4 : 1) * VLENB / sizeof(T);
  for (int i = 0; i < size; i++) {
    dut[i] = static_cast<T>(i);
  }

  if (stripmine) {
    dut = vldpm_xx(dut, length);
  } else {
    dut = vldp_xx(dut, length);
  }

  void* target = reinterpret_cast<void*>(read_data + (length * sizeof(T)));
  if (reinterpret_cast<void*>(dut) != target) {
    printf("Failed test_ldst_p_xx for type %s:\n", TypeString<T>::type_str());
    printf("  expected dut=%x to be %x\n",
           reinterpret_cast<void*>(dut), target);
    exit(1);
  }

  uint8_t write_data[4*VLENB] __attribute__((aligned(64)));
  T* result = reinterpret_cast<T*>(write_data);
  if (stripmine) {
    vst_x_m(T, v0, result);
  } else {
    vst_x(T, v0, result);
  }

  for (int i = 0; i < size; i++) {
    if (result[i] != static_cast<T>(i)) {
      printf("Failed test_ldst_p_xx for type %s:\n", TypeString<T>::type_str());
      printf("  expected %d at position %d got %d\n", i, i, result[i]);
      exit(1);
    }
  }
}

// Test vld_l[p][_m] instructions for correct pointer update and load result.
template<typename T, bool post_increment, bool stripmine>
void test_ldst_l_xx(int length) {
  uint8_t read_data[4*VLENB] __attribute__((aligned(64)));
  T* dut = reinterpret_cast<T*>(read_data);
  constexpr int size = (stripmine ? 4 : 1) * VLENB / sizeof(T);
  for (int i = 0; i < size; i++) {
    dut[i] = static_cast<T>(i);
  }
  // Saturate length
  length = std::min(size, length);

  if (post_increment) {
    if (stripmine) {
      dut = vldlpm_data(dut, length);
    } else {
      dut = vldlp_data(dut, length);
    }
  } else {
    if (stripmine) {
      vld_l_xx_m(T, v0, dut, length);
    } else {
      vld_l_xx(T, v0, dut, length);
    }
  }

  void* target = reinterpret_cast<void*>(read_data +
        (post_increment ? length * sizeof(T) : 0));
  if (reinterpret_cast<void*>(dut) != target) {
    printf("Failed test_ldst_l_xx for type %s:\n", TypeString<T>::type_str());
    printf("  expected dut=%x to be %x\n",
           reinterpret_cast<void*>(dut), target);
    exit(1);
  }

  uint8_t write_data[4*VLENB] __attribute__((aligned(64)));
  T* result = reinterpret_cast<T*>(write_data);
  if (stripmine) {
    vst_x_m(T, v0, result);
  } else {
    vst_x(T, v0, result);
  }

  int i = 0;
  for (; i < length; i++) {
    if (result[i] != static_cast<T>(i)) {
      printf("Failed test_ldst_l_xx for type %s:\n", TypeString<T>::type_str());
      printf("  expected %d at position %d/%d got %d\n", i, i, size, result[i]);
      exit(1);
    }
  }
  for (; i < size; i++) {
    if (result[i] != 0) {
      printf("Failed test_ldst_l_xx for type %s:\n", TypeString<T>::type_str());
      printf("  expected 0 at position %d/%d got %d\n", i, size, result[i]);
      exit(1);
    }
  }
}

// Test vld_s[p][_m] instructions for correct pointer update and load result.
template<typename T, bool post_increment, bool stripmine>
void test_ldst_s_xx(int stride) {
  uint8_t read_data[9*VLENB] __attribute__((aligned(64)));
  T* dut = reinterpret_cast<T*>(read_data);
  for (uint32_t i = 0; i < 9 * VLENB / sizeof(T); i++) {
    dut[i] = static_cast<T>(i);
  }

  if (post_increment) {
    if (stripmine) {
      dut = vldspm_xx(dut, stride);
    } else {
      dut = vldsp_xx(dut, stride);
    }
  } else {
    if (stripmine) {
      vld_s_xx_m(T, v0, dut, stride);
    } else {
      vld_s_xx(T, v0, dut, stride);
    }
  }

  int offset = 0;
  if (post_increment) {
    offset = (stripmine ? 4 * stride : stride) * sizeof(T);
  }
  void* target = reinterpret_cast<void*>(read_data + offset);
  if (reinterpret_cast<void*>(dut) != target) {
    printf("Failed test_ldst_s_xx for type %s:\n", TypeString<T>::type_str());
    printf("  expected dut=%x to be %x\n",
           reinterpret_cast<void*>(dut), target);
    exit(1);
  }

  uint8_t write_data[4*VLENB] __attribute__((aligned(64)));
  T* result = reinterpret_cast<T*>(write_data);
  if (stripmine) {
    vst_x_m(T, v0, result);
  } else {
    vst_x(T, v0, result);
  }

  constexpr int num_vectors = stripmine ? 4 : 1;
  for (int v = 0; v < num_vectors; v++) {
    const T* result_vector = result + (v * VLENB / sizeof(T));
    int start_value = v * stride;
    for (uint32_t i = 0; i < VLENB / sizeof(T); i++) {
      T expected_value = static_cast<T>(start_value + i);
      if (result_vector[i] != start_value + i) {
        printf("Failed test_ldst_s_xx for type %s:\n",
               TypeString<T>::type_str());
        printf("  expected %d at position %d got %d\n",
               expected_value, i, result_vector[i]);
        exit(1);
      }
    }
  }
}

// Test vld_tp[_m] instructions for correct pointer update and load result.
template<typename T, bool stripmine>
void test_ldst_tp_xx(int stride) {
  uint8_t read_data[9*VLENB] __attribute__((aligned(64)));
  T* dut = reinterpret_cast<T*>(read_data);
  for (uint32_t i = 0; i < 9 * VLENB / sizeof(T); i++) {
    dut[i] = static_cast<T>(i);
  }

  if (stripmine) {
    dut = vldtpm_xx(dut, stride);
  } else {
    dut = vldtp_xx(dut, stride);
    stride = std::min(stride, static_cast<int>(VLENB / sizeof(T)));
  }

  void* target = reinterpret_cast<void*>(read_data + VLENB);
  if (reinterpret_cast<void*>(dut) != target) {
    printf("Failed test_ldst_tp_xx for type %s:\n", TypeString<T>::type_str());
    printf("  expected dut=%x to be %x\n",
           reinterpret_cast<void*>(dut), target);
    exit(1);
  }

  uint8_t write_data[4*VLENB] __attribute__((aligned(64)));
  T* result = reinterpret_cast<T*>(write_data);
  if (stripmine) {
    vst_x_m(T, v0, result);
  } else {
    vst_x(T, v0, result);
  }

  int i = 0;
  for (; i < stride; i++) {
    int v = i / (VLENB / sizeof(T));
    int r = i % (VLENB / sizeof(T));
    T expected_value = static_cast<T>(v * stride + r);

    if (result[i] != expected_value) {
      printf("Failed test_ldst_tp_xx for type %s:\n",
             TypeString<T>::type_str());
      printf("  expected %d at position %d got %d\n",
             expected_value, i, result[i]);
      exit(1);
    }
  }
  constexpr int num_vectors = stripmine ? 4 : 1;
  for (; i < static_cast<int>(num_vectors * VLENB / sizeof(T)); i++) {
    if (result[i] != 0) {
      printf("Failed test_ldst_tp_xx for type %s:\n",
             TypeString<T>::type_str());
      printf("  expected 0 at position %d got %d\n", i, result[i]);
      exit(1);
    }
  }
}

// Test vld_l instructions.
template <typename T, bool sm>
void test_vld_l() {
  constexpr int size = 4 * VLENB / sizeof(T);
  T in[size] __attribute__((aligned(64)));
  T dut[size] __attribute__((aligned(64)));
  int vlen;
  if (sm) {
    getmaxvl_m(T, vlen);
  } else {
    getmaxvl(T, vlen);
  }

  for (int j = 0; j <= vlen; ++j) {
    in[j] = j + 1;
  }

  for (int i = 0; i <= vlen; ++i) {
    if (sm) {
      vld_l_xx_m(T, v0, in, i);
      vst_x_m(T, v0, dut);
    } else {
      vld_l_xx(T, v0, in, i);
      vst_x(T, v0, dut);
    }
    for (int j = 0; j < vlen; ++j) {
      T ref = j < i ? j + 1 : 0;
      if (ref != dut[j]) {
        printf("**error vld_l[%d,%d] %x %x\n", i, j, ref, dut[j]);
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

  // vld.*.x
  test_ldst_x<uint8_t, false, false>();
  test_ldst_x<uint16_t, false, false>();
  test_ldst_x<uint32_t, false, false>();

  // vld.*.p.x
  test_ldst_x<uint8_t, true, false>();
  test_ldst_x<uint16_t, true, false>();
  test_ldst_x<uint32_t, true, false>();

  // vld.*.p.x.m
  test_ldst_x<uint8_t, true, true>();
  test_ldst_x<uint16_t, true, true>();
  test_ldst_x<uint32_t, true, true>();

  test_ldst_l_xx<uint8_t, false, false>(0);
  test_ldst_l_xx<uint16_t, false, false>(0);
  test_ldst_l_xx<uint32_t, false, false>(0);
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

  test_ldst_s_xx<uint8_t, false, false>(0);
  test_ldst_s_xx<uint16_t, false, false>(0);
  test_ldst_s_xx<uint32_t, false, false>(0);
  test_ldst_s_xx<uint8_t, false, false>(1);
  test_ldst_s_xx<uint16_t, false, false>(1);
  test_ldst_s_xx<uint32_t, false, false>(1);
  test_ldst_s_xx<uint8_t, false, false>(vlenb / 2);
  test_ldst_s_xx<uint16_t, false, false>(vlenh / 2);
  test_ldst_s_xx<uint32_t, false, false>(vlenw / 2);
  test_ldst_s_xx<uint8_t, false, false>(vlenb);
  test_ldst_s_xx<uint16_t, false, false>(vlenh);
  test_ldst_s_xx<uint32_t, false, false>(vlenw);
  test_ldst_s_xx<uint8_t, false, false>(vlenb + 1);
  test_ldst_s_xx<uint16_t, false, false>(vlenh + 1);
  test_ldst_s_xx<uint32_t, false, false>(vlenw + 1);

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

  test_ldst_s_xx<uint8_t, true, false>(0);
  test_ldst_s_xx<uint16_t, true, false>(0);
  test_ldst_s_xx<uint32_t, true, false>(0);
  test_ldst_s_xx<uint8_t, true, false>(1);
  test_ldst_s_xx<uint16_t, true, false>(1);
  test_ldst_s_xx<uint32_t, true, false>(1);
  test_ldst_s_xx<uint8_t, true, false>(vlenb / 2);
  test_ldst_s_xx<uint16_t, true, false>(vlenh / 2);
  test_ldst_s_xx<uint32_t, true, false>(vlenw / 2);
  test_ldst_s_xx<uint8_t, true, false>(vlenb);
  test_ldst_s_xx<uint16_t, true, false>(vlenh);
  test_ldst_s_xx<uint32_t, true, false>(vlenw);
  test_ldst_s_xx<uint8_t, true, false>(vlenb + 1);
  test_ldst_s_xx<uint16_t, true, false>(vlenh + 1);
  test_ldst_s_xx<uint32_t, true, false>(vlenw + 1);

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
  test_ldst_p_xx<uint8_t, false>(1024);
  test_ldst_p_xx<uint16_t, false>(1024);
  test_ldst_p_xx<uint32_t, false>(1024);

  test_ldst_tp_xx<uint8_t, false>(1);
  test_ldst_tp_xx<uint16_t, false>(1);
  test_ldst_tp_xx<uint32_t, false>(1);
  test_ldst_tp_xx<uint8_t, false>(vlenb / 2);
  test_ldst_tp_xx<uint16_t, false>(vlenh / 2);
  test_ldst_tp_xx<uint32_t, false>(vlenw / 2);
  test_ldst_tp_xx<uint8_t, false>(vlenb - 1);
  test_ldst_tp_xx<uint16_t, false>(vlenh - 1);
  test_ldst_tp_xx<uint32_t, false>(vlenw - 1);
  test_ldst_tp_xx<uint8_t, false>(vlenb);
  test_ldst_tp_xx<uint16_t, false>(vlenh);
  test_ldst_tp_xx<uint32_t, false>(vlenw);
  test_ldst_tp_xx<uint8_t, false>(vlenb + 1);
  test_ldst_tp_xx<uint16_t, false>(vlenh + 1);
  test_ldst_tp_xx<uint32_t, false>(vlenw + 1);

  // strip mine only
  test_ldst_l_xx<uint8_t, false, true>(0);
  test_ldst_l_xx<uint16_t, false, true>(0);
  test_ldst_l_xx<uint32_t, false, true>(0);
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

  test_ldst_s_xx<uint8_t, false, true>(0);
  test_ldst_s_xx<uint16_t, false, true>(0);
  test_ldst_s_xx<uint32_t, false, true>(0);
  test_ldst_s_xx<uint8_t, false, true>(1);
  test_ldst_s_xx<uint16_t, false, true>(1);
  test_ldst_s_xx<uint32_t, false, true>(1);
  test_ldst_s_xx<uint8_t, false, true>(vlenb / 2);
  test_ldst_s_xx<uint16_t, false, true>(vlenh / 2);
  test_ldst_s_xx<uint32_t, false, true>(vlenw / 2);
  test_ldst_s_xx<uint8_t, false, true>(vlenb);
  test_ldst_s_xx<uint16_t, false, true>(vlenh);
  test_ldst_s_xx<uint32_t, false, true>(vlenw);
  test_ldst_s_xx<uint8_t, false, true>(vlenb + 1);
  test_ldst_s_xx<uint16_t, false, true>(vlenh + 1);
  test_ldst_s_xx<uint32_t, false, true>(vlenw + 1);

  test_ldst_p_xx<uint8_t, true>(1);
  test_ldst_p_xx<uint16_t, true>(1);
  test_ldst_p_xx<uint32_t, true>(1);
  test_ldst_p_xx<uint8_t, true>(vlenb / 2);
  test_ldst_p_xx<uint16_t, true>(vlenh / 2);
  test_ldst_p_xx<uint32_t, true>(vlenw / 2);
  test_ldst_p_xx<uint8_t, true>(vlenb - 1);
  test_ldst_p_xx<uint16_t, true>(vlenh - 1);
  test_ldst_p_xx<uint32_t, true>(vlenw - 1);
  test_ldst_p_xx<uint8_t, true>(vlenb);
  test_ldst_p_xx<uint16_t, true>(vlenh);
  test_ldst_p_xx<uint32_t, true>(vlenw);
  test_ldst_p_xx<uint8_t, true>(1024);
  test_ldst_p_xx<uint16_t, true>(1024);
  test_ldst_p_xx<uint32_t, true>(1024);

  // post increment, stripmine
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

  test_ldst_s_xx<uint8_t, true, true>(0);
  test_ldst_s_xx<uint16_t, true, true>(0);
  test_ldst_s_xx<uint32_t, true, true>(0);
  test_ldst_s_xx<uint8_t, true, true>(1);
  test_ldst_s_xx<uint16_t, true, true>(1);
  test_ldst_s_xx<uint32_t, true, true>(1);
  test_ldst_s_xx<uint8_t, true, true>(vlenb / 2);
  test_ldst_s_xx<uint16_t, true, true>(vlenh / 2);
  test_ldst_s_xx<uint32_t, true, true>(vlenw / 2);
  test_ldst_s_xx<uint8_t, true, true>(vlenb);
  test_ldst_s_xx<uint16_t, true, true>(vlenh);
  test_ldst_s_xx<uint32_t, true, true>(vlenw);
  test_ldst_s_xx<uint8_t, true, true>(vlenb + 1);
  test_ldst_s_xx<uint16_t, true, true>(vlenh + 1);
  test_ldst_s_xx<uint32_t, true, true>(vlenw + 1);

  test_ldst_tp_xx<uint8_t, true>(1);
  test_ldst_tp_xx<uint16_t, true>(1);
  test_ldst_tp_xx<uint32_t, true>(1);
  test_ldst_tp_xx<uint8_t, true>(vlenb / 2);
  test_ldst_tp_xx<uint16_t, true>(vlenh / 2);
  test_ldst_tp_xx<uint32_t, true>(vlenw / 2);
  test_ldst_tp_xx<uint8_t, true>(vlenb - 1);
  test_ldst_tp_xx<uint16_t, true>(vlenh - 1);
  test_ldst_tp_xx<uint32_t, true>(vlenw - 1);
  test_ldst_tp_xx<uint8_t, true>(vlenb);
  test_ldst_tp_xx<uint16_t, true>(vlenh);
  test_ldst_tp_xx<uint32_t, true>(vlenw);
  test_ldst_tp_xx<uint8_t, true>(vlenb + 1);
  test_ldst_tp_xx<uint16_t, true>(vlenh + 1);
  test_ldst_tp_xx<uint32_t, true>(vlenw + 1);

  test_vld_l<uint8_t, 0>();
  test_vld_l<uint16_t, 0>();
  test_vld_l<uint32_t, 0>();

  test_vld_l<uint8_t, 1>();
  test_vld_l<uint16_t, 1>();
  test_vld_l<uint32_t, 1>();

  return 0;
}
