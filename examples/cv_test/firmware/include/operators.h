#pragma once

#include <cstdint>
#include "common.h"

namespace kelvin_cv {
int op_yu12_to_yv12(const InputHeader& in_hdr, uint8_t* in, uint8_t* out, std::size_t &out_len);
int op_yu12_to_nv12(const InputHeader& in_hdr, uint8_t* in, uint8_t* out, std::size_t &out_len);
int op_nv12_to_nv21(const InputHeader& in_hdr, uint8_t* in, uint8_t* out, std::size_t &out_len);
int op_convertScaleAbs(const InputHeader& in_hdr, uint8_t* in, uint8_t* out, std::size_t &out_len);
int op_sobel(const InputHeader& in_hdr, uint8_t* in, uint8_t* out, std::size_t &out_len);
} // namespace kelvin_cv


