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

int op_sobel(const InputHeader& in_hdr, uint8_t* in, uint8_t* out, std::size_t &out_len)
{
    /* 0. 加载X和Y方向的卷积核 */
    /* 1. 加载一个像素及其3x3领域：考虑padding */
    /* 2. 转换为16位整型 */
    /* 3. 计算X和Y方向的卷积 */
    return 0;
}

}
