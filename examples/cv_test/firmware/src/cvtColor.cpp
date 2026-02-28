#include "../include/common.h"
#include "crt/kelvin.h"
#include <cstddef>

namespace kelvin_cv {

int op_yu12_to_yv12(const InputHeader& in_hdr, uint8_t* in, uint8_t* out, std::size_t &out_len)
{
    uint32_t vl = 0;
    uint32_t n = 0;
    /* Y plane length */
    const uint32_t Y_len = in_hdr.width * in_hdr.height;
    /* U plane or V plane length */
    const uint32_t UV_len = (in_hdr.width * in_hdr.height)/4;

    n = Y_len;
    do {
        getvl_b_x_m(vl, n);
        n -= vl;
        vld_b_lp_xx_m(vm1, in, vl);
        vst_b_lp_xx_m(vm1, out, vl);
    } while( n > 0);

    n = UV_len;
    /* input increase to input + Y_len, so do output */
    uint8_t *input_U_plane_base = in;
    uint8_t *input_V_plane_base = in + UV_len;
    uint8_t *output_U_plane_base = out + UV_len;
    uint8_t *output_V_plane_base = out;
    do {
        getvl_b_x_m(vl, n);
        n -= vl;
        vld_b_lp_xx_m(vm1, input_U_plane_base, vl);
        vld_b_lp_xx_m(vm2, input_V_plane_base, vl);
        vst_b_lp_xx_m(vm1, output_U_plane_base, vl);
        vst_b_lp_xx_m(vm2, output_V_plane_base, vl);
    } while(n > 0);

    out_len = Y_len + UV_len*2;

    return 0;
}

int op_yu12_to_nv12(const InputHeader& in_hdr, uint8_t* in, uint8_t* out, std::size_t &out_len)
{
    uint32_t vl = 0;
    uint32_t n = 0;
    /* Y plane length */
    const uint32_t Y_len = in_hdr.width * in_hdr.height;
    /* U plane or V plane length */
    const uint32_t UV_len = (in_hdr.width * in_hdr.height)/4;

    /* copy Y plane */
    n = Y_len;
    do {
        getvl_b_x_m(vl, n);
        n -= vl;
        vld_b_lp_xx_m(vm1, in, vl);
        vst_b_lp_xx_m(vm1, out, vl);
    } while( n > 0);

    n = UV_len;
    /* input increase to input + Y_len, so do output */
    uint8_t *input_U_plane_base = in;
    uint8_t *input_V_plane_base = in + UV_len;
    uint8_t *output_UV_base = out;
#if 0 /* normal test */
    getmaxvl_b(vl);
    std::size_t simd_cycles = UV_len / (vl*2);
    std::size_t scalar_cycles = UV_len % (vl*2);
    for (uint32_t c=0; c < simd_cycles; c++) {
        vld_b_p_x(v0, input_U_plane_base);
        vld_b_p_x(v1, input_V_plane_base);
        vld_b_p_x(v2, input_U_plane_base);
        vld_b_p_x(v3, input_V_plane_base);
        vzip_b_vv(v4, v0, v1);
        vzip_b_vv(v6, v2, v3);

        vst_b_p_x_m(vm1, output_UV_base);
    }
#else /* stripmine test */
    getmaxvl_b_m(vl);
    std::size_t simd_cycles = UV_len / (vl);
    std::size_t scalar_cycles = UV_len % (vl);
    for (uint32_t c=0; c < simd_cycles; c++) {
        vld_b_p_x_m(vm0, input_U_plane_base);
        vld_b_p_x_m(vm1, input_V_plane_base);

        vzip_b_vv_m(vm4, vm0, vm1);

        vst_b_p_x_m(vm4, output_UV_base);
        vst_b_p_x_m(vm5, output_UV_base);
    }
#endif
    for (uint32_t i = 0; i < scalar_cycles; i++) {
        *(output_UV_base + 2*i) = *(input_U_plane_base + i);
        *(output_UV_base + 2*i + 1) = *(input_V_plane_base + i);
    }

    out_len = Y_len + UV_len*2;

    return 0;
}

}
