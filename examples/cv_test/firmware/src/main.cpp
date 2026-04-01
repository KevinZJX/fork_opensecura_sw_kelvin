#include <cstdint>
#include <iostream>

#include "crt/kelvin.h"
#include "../include/common.h"
#include "../include/memory_layout.h"
#include "../include/operators.h"

extern "C" {
#include <stdio.h>
}

using namespace kelvin_cv;

__attribute__((section(".model_output_header")))
OutputHeader_t g_out_hdr = {0};

int main(int argc, char **argv)
{
    InputHeader_t *in_hdr = (InputHeader_t*)get_model_input_header_buffer();
    uint8_t *in_buf = (uint8_t*)get_model_input_buffer();
    uint8_t *out_buf = (uint8_t*)get_model_output_buffer();
    std::size_t out_len = 0;
    uint32_t ret_status;

    printf("In h:%d w:%d\n", in_hdr->height, in_hdr->width);

    switch(in_hdr->opcode) {
    case OP_YU12_TO_YV12:
    case OP_YV12_TO_YU12:
        ret_status = op_yu12_to_yv12(*in_hdr, in_buf, out_buf, out_len);
        break;
    case OP_YU12_TO_NV12:
        ret_status = op_yu12_to_nv12(*in_hdr, in_buf, out_buf, out_len);
        break;
    case OP_NV12_TO_NV21:
        ret_status = op_nv12_to_nv21(*in_hdr, in_buf, out_buf, out_len);
        break;
    case OP_CVTSCALEABS:
        ret_status = op_convertScaleAbs(*in_hdr, in_buf, out_buf, out_len);
        break;
    case OP_THRESHOLD:
        ret_status = op_threshold(*in_hdr, in_buf, out_buf, out_len);
        break;
    case OP_SOBEL:
        ret_status = op_sobel(*in_hdr, in_buf, out_buf, out_len);
        break;
    default:
        ret_status = -1;
        break;
    }

    printf("Output size:%ld\n", out_len);
    g_out_hdr.length = out_len;
    g_out_hdr.cycles = 0;
    g_out_hdr.status = ret_status;

    return 0;
}

