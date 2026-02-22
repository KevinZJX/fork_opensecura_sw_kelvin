// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// A Simple kelvin program.

#include <stddef.h>
#include <stdint.h>
#include <stdio.h>

// This header is included to ensure that it can
// be referenced from a C file.
#include "crt/kelvin.h"
#include "memory_layout.h"

typedef struct {
  uint32_t return_code;  // Populated in kelvin_start.S.
  uint32_t output_ptr;
  uint32_t length;
} OutputHeader;

typedef struct {
    uint32_t width;
    uint32_t height;
    uint32_t channels;
    uint32_t color_type;
} ImageProperty;

typedef struct {
    uint32_t nImages;
    ImageProperty images[];
} InputHeader;

__attribute__((section(".model_output_header"))) OutputHeader output_header = {
	.output_ptr = 0,
	.length = 0,
};

size_t get_image_size(ImageProperty *prop)
{
    return (size_t)(prop->width * prop->height * prop->channels);
}

int main(int argc, char *argv[]) {
	uint8_t *output = NULL;
	uint8_t *input = NULL;
    InputHeader *input_header = NULL;
    uint32_t input_len = 0;
    uint32_t vl = 0;

	/* initialize input and output buffer */
	output = get_model_output_buffer();
	input = get_model_input_buffer();
    input_header = (InputHeader*)get_model_input_header_buffer();

    input_len = get_image_size(&input_header->images[0]);

    /**
     * @brief Exchange YUV420P U and V plane
     * for example: YU12 to YV12
     * YV12 to YU12
     */
    const uint32_t Y_len = input_header->images[0].width * input_header->images[0].height;
    /* U plane or V plane length */
    const uint32_t UV_len = (input_header->images[0].width * input_header->images[0].height)/4;
    printf("Y_len: %d UV_len:%d\n", Y_len, UV_len);
    uint32_t n = Y_len;

    do {
        getvl_b_x_m(vl, n);
        n -= vl;
        vld_b_lp_xx_m(vm1, input, vl);
        vst_b_lp_xx_m(vm1, output, vl);
    } while( n > 0);

    n = UV_len;
    /* input increase to input + Y_len, so do output */
    uint8_t *input_V_plane_base = input;
    uint8_t *input_U_plane_base = input + UV_len;
    uint8_t *output_U_plane_base = output;
    uint8_t *output_V_plane_base = output + UV_len;
    do {
        getvl_b_x_m(vl, n);
        n -= vl;
        vld_b_lp_xx_m(vm1, input_U_plane_base, vl);
        vld_b_lp_xx_m(vm2, input_V_plane_base, vl);
        vst_b_lp_xx_m(vm1, output_U_plane_base, vl);
        vst_b_lp_xx_m(vm2, output_V_plane_base, vl);
    } while(n > 0);

	// Setup output.
    output_header.length = Y_len + UV_len*2;
    printf("output: %d\n", output_header.length);
	output_header.output_ptr = (uint32_t)output;

	return 0;
}
