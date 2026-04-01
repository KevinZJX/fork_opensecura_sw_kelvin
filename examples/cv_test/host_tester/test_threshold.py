import cv2 as cv
import os
import numpy as np
import struct
from kelvin_tester import KelvinTester

opensecura_root = "/home/xu/workspace/opensecura/source"
opencv_root = "/home/xu/workspace/opencv/opencv-4.12.0/"

def float_to_uint32(f):
    # '<f' 表示单精度浮点数（小端）
    # '<I' 表示无符号整数（小端）
    # 将 float 打包成字节流，再按无符号整数解析出来
    return struct.unpack('<I', struct.pack('<f', f))[0]

def run_kelvin_operator(tester, image_path):
    # 准备测试数据 (YU12)
    img_bgr = cv.imread(opencv_root + image_path)
    img_yuv = cv.cvtColor(img_bgr, cv.COLOR_BGR2YUV_I420) # YU12
    h, w = img_bgr.shape[0], img_bgr.shape[1]

    threshould = 128
    params = [0]*10
    params[0] = threshould

    # 执行算子 (OP_YU12_TO_NV12 = 3)
    status, out_len, yu12_bytes, cycles = tester.run_kernel(14, img_yuv.tobytes(), w, h, params=params, timeout=10)

    if status == 0:
        # 将结果转回 BGR 验证
        yu12_data = np.frombuffer(yu12_bytes, dtype=np.uint8).reshape(h*3//2, w)
        res_bgr = cv.cvtColor(yu12_data, cv.COLOR_YUV2GRAY_I420)
        cv.imshow("Result", res_bgr)
        cv.waitKey(0)
        print(f"Success! Performance: {cycles} cycles")

def main():
    # 切换到Opensecura项目根目录
    try:
        os.chdir(opensecura_root)
    except FileNotFoundError:
        print(f"Error: Dir {opensecura_root} not exist")
    except PermissionError:
        print(f"Error: No Permission in {opensecura_root}")
    except Exception as e:
        print(f"Unknown error: {e}")
    # 初始化框架
    tester = KelvinTester(f"{opensecura_root}/sw/kelvin/bazel-out/k8-fastbuild-ST-97eccf989747/bin/examples/cv_test/firmware/kelvin_firmware.bin", "sim/config/kelvin.resc")
    # 执行测试
    run_kelvin_operator(tester, "samples/data/lena.jpg")

if __name__ == "__main__":
    main()
