#include "../include/common.h"
#include "crt/kelvin.h"
#include <cstddef>

namespace kelvin_cv {

int op_convertScaleAbs(const InputHeader& in_hdr, uint8_t* in, uint8_t* out, std::size_t &out_len)
{
    uint32_t vl = 0;
    uint32_t n = 0;
    /* Y plane length */
    const uint32_t Y_len = in_hdr.width * in_hdr.height;
    /* U plane or V plane length */
    const uint32_t UV_len = (in_hdr.width * in_hdr.height)/4;

    int constrast_ratio = (int)in_hdr.params[0].i;
    int brightness_ratio = (int)in_hdr.params[1].i;

    /* cvt Y plane */
    n = Y_len;
    do {
        getvl_b_x_m(vl, n);
        n -= vl;
        vld_b_lp_xx_m(vm0, in, vl);
        vaddw_h_u_vx_m(vm2, vm0, 0);
        /* 调节Y通道的值就可以进行亮度调节 */
        vmuls_h_u_vx_m(vm0, vm2, constrast_ratio);
        vmuls_h_u_vx_m(vm1, vm3, constrast_ratio);
        vsransu_b_r_vx_m(vm2, vm0, 0x8);
        vadds_b_u_vx_m(vm1, vm2, brightness_ratio);
        vst_b_lp_xx_m(vm1, out, vl);
    } while( n > 0);

    n = UV_len;
    /* input increase to input + Y_len, so do output */
    /* copy UV plane */
    uint8_t *input_U_plane_base = in;
    uint8_t *input_V_plane_base = in + UV_len;
    uint8_t *output_U_plane_base = out;
    uint8_t *output_V_plane_base = out + UV_len;
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

/* 给定全局阈值处理二值化图像 */
int op_threshold(const InputHeader& in_hdr, uint8_t* in, uint8_t* out, std::size_t &out_len)
{
    uint32_t vl = 0;
    uint32_t n = 0;
    /* Y plane length */
    const uint32_t Y_len = in_hdr.width * in_hdr.height;
    /* U plane or V plane length */
    const uint32_t UV_len = (in_hdr.width * in_hdr.height)/4;

    int threshold = (int)in_hdr.params[0].i;

    /* cvt Y plane */
    n = Y_len;
    do {
        getvl_b_x_m(vl, n);
        n -= vl;
        vld_b_lp_xx_m(vm0, in, vl);
        vge_b_u_vx_m(vm1, vm0, threshold);
        vdup_b_x_m(vm0, 255);
        vsel_b_vx_m(vm0, vm1, 0);
        vst_b_lp_xx_m(vm0, out, vl);
    } while( n > 0);

    n = UV_len;
    /* input increase to input + Y_len, so do output */
    /* copy UV plane */
    uint8_t *input_U_plane_base = in;
    uint8_t *input_V_plane_base = in + UV_len;
    uint8_t *output_U_plane_base = out;
    uint8_t *output_V_plane_base = out + UV_len;
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

int op_blur(const InputHeader& in_hdr, uint8_t* in, uint8_t* out, std::size_t &out_len)
{
    uint32_t vl1 = 0;
    uint32_t vl2 = 0;
    uint32_t n = 0;
    /* Y plane length */
    const uint32_t Y_len = in_hdr.width * in_hdr.height;
    /* U plane or V plane length */
    const uint32_t UV_len = (in_hdr.width * in_hdr.height)/4;

    int threshold = (int)in_hdr.params[0].i;

    /* cvt Y plane */
    n = Y_len;
    // vm0 是本次计算的数据
    // vm1 是上一次计算的数据
    // vm2 是下一次计算的数据
    // vm3 是左邻区数据
    // vm4 是右邻区数据
    vdup_b_x_m(vm1, 0);
    getvl_b_x_m(vl1, n);
    n -= vl1;
    vld_b_lp_xx_m(vm0, in, vl1);

    do {
        if (n > 0) {
            getvl_b_x_m(vl2, n);
            n -= vl2;
            vld_b_lp_xx_m(vm2, in, vl2);
        } else {
            vdup_b_x_m(vm2, 0);
            vl2 = 0;
        }
        vslidehp_b_1_vv_m(vm3, vm1, vm0);
        vslidehn_b_1_vv_m(vm4, vm0, vm2);
        vaddw_h_u_vv_m(vm5, vm0, vm3);
        vaddw_h_u_vx_m(vm7, vm4, 0);
        vadds_h_u_vv_m(vm9, vm5, vm7);
        vadds_h_u_vv_m(vm10, vm6, vm8);
        vmuls_h_u_vx_m(vm5, vm9, 85);
        vmuls_h_u_vx_m(vm6, vm10, 85);
        vsransu_b_r_vx_m(vm7, vm5, 8);
        vst_b_lp_xx_m(vm7, out, vl1);
        vmv_v_m(vm1, vm0);
        vmv_v_m(vm0, vm2);
        vl1 = vl2;
    } while(vl1 > 0);

    n = UV_len;
    /* input increase to input + Y_len, so do output */
    /* copy UV plane */
    uint8_t *input_U_plane_base = in;
    uint8_t *input_V_plane_base = in + UV_len;
    uint8_t *output_U_plane_base = out;
    uint8_t *output_V_plane_base = out + UV_len;
    do {
        getvl_b_x_m(vl1, n);
        n -= vl1;
        vld_b_lp_xx_m(vm1, input_U_plane_base, vl1);
        vld_b_lp_xx_m(vm2, input_V_plane_base, vl1);
        vst_b_lp_xx_m(vm1, output_U_plane_base, vl1);
        vst_b_lp_xx_m(vm2, output_V_plane_base, vl1);
    } while(n > 0);

    out_len = Y_len + UV_len*2;

    return 0;
}

int op_sobel(const InputHeader& in_hdr, uint8_t* in, uint8_t* out, std::size_t &out_len)
{
    /* 0. 加载X和Y方向的卷积核 */
    /* 1. 加载一个像素及其3x3领域：考虑padding */
    /* 2. 转换为16位整型 */
    /* 3. 计算X和Y方向的卷积 */
    return 0;
}

}
