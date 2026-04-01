#pragma once
#include <cstdint>

namespace kelvin_cv {

enum Opcode : uint32_t {
    OP_COPY         = 0,
    OP_YU12_TO_YV12 = 1,
    OP_YV12_TO_YU12,
    OP_YU12_TO_NV12,
    OP_YV12_TO_NV12,
    OP_YU12_TO_NV21,
    OP_YV12_TO_NV21,
    OP_NV12_TO_NV21,
    OP_NV12_TO_YU12,
    OP_NV12_TO_YV12,
    OP_NV21_TO_NV12,
    OP_NV21_TO_YU12,
    OP_NV21_TO_YV12,
    OP_CVTSCALEABS,
    OP_THRESHOLD,
    OP_SOBEL,
    OP_CONV2D,
};

union Arg {
    uint32_t u;
    int32_t i;
    float f;
};

typedef struct InputHeader {
    uint32_t opcode;
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t stride;        // 图像对齐步长
    uint32_t format;        // 预留图像格式字段
    Arg      params[10];    // 给不同算子的参数
} InputHeader_t;

typedef struct OutputHeader {
    uint32_t status;
    uint32_t length;
    uint32_t cycles;
    uint32_t reserved;
} OutputHeader_t;


} // namespace kelvin_cv
