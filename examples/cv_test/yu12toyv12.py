import numpy as np
import struct
from pyrenode3.wrappers import Emulation, Monitor
from System import Array, Byte
import time
import cv2 as cv
import os

opensecura_root = "/home/xu/workspace/opensecura/source"
opencv_root = "/home/xu/workspace/opencv/opencv-4.12.0/"

model_input_addr = 0x5A800000
model_output_addr = 0x5A400000
model_input_header_addr = 0x5AFFFF80
model_output_header_addr = 0x5AFFFFC0

def read_model_output(bus, len):
    ret = bus.ReadDoubleWord(model_output_header_addr)
    output_len = bus.ReadDoubleWord(model_output_header_addr + 0x8)
    result_bytes = bus.ReadBytes(model_output_addr, output_len)
    return ret, output_len, result_bytes

def write_model_input(bus, data, n_images, image_props):
    # Write row data into input buffer
    print(f"Python wrote {data.Length} bytes row data\n")
    bus.WriteBytes(model_input_addr, data, 0, data.Length)
    # Write images informations into input header buffer
    input_buffer = bytearray(struct.pack("<I", n_images))

    for prop in image_props:
        w, h, c, t = prop
        packed_prop = struct.pack("<IIII", w, h, c, t)
        input_buffer.extend(packed_prop)

    bus.WriteBytes(model_input_header_addr, input_buffer, 0, len(input_buffer))

def get_opencv_samples(image_path):
    return cv.imread(opencv_root + image_path)


def run_kelvin_operator(image_path):
    m = Monitor()
    e = Emulation()

    # 1. 动态注入二进制文件，因为必须要在加载kelvin.resc之前覆盖$bin变量的值，所以使用此方法
    m.Parse(f"$bin=@{opensecura_root}/sw/kelvin/bazel-out/k8-fastbuild-ST-97eccf989747/bin/examples/cv_test/yu12toyv12.bin")

    # 2. 加载仿真环境
    if not m.Parse("i @sim/config/kelvin.resc"):
        print("Failed to load kelvin.resc!")
        return

    # 3. Get simulation instance
    kelvin_mach = e.get_mach("kelvin")
    sysbus = kelvin_mach.sysbus

    # Prepare image data
    lena_jpg = get_opencv_samples(image_path)
    lena_yv12 = cv.cvtColor(lena_jpg, cv.COLOR_BGR2YUV_YV12)
    lena_data_flatten = lena_yv12.flatten()
    dotnet_data = Array[Byte](lena_yv12.tobytes())

    #  TODO: set according to images property
    image_list = [
        (512, 512, 3, 0),
    ]

    write_model_input(sysbus, dotnet_data, n_images=len(image_list), image_props=image_list)
    # 启动仿真
    e.StartAll()
    # 释放Kelvin-CPU
    sysbus.ml_top_controlblock.WriteDoubleWord(0xc, 0)

    while False == sysbus.cpu2.IsHalted:
        time.sleep(0.01)

    ret_code, output_len, result_bytes = read_model_output(sysbus, 0x40)

    lena_received = np.frombuffer(result_bytes, count=output_len, dtype=np.uint8)

    lena_received_bgr = cv.cvtColor(lena_received.reshape(512*3//2, 512), cv.COLOR_YUV2BGR_I420)

    cv.imshow("lena", lena_received_bgr)
    cv.waitKey(0)

    print(lena_data_flatten.shape, lena_received.shape)

    if output_len == dotnet_data.Length:
        print(f"Test Passed \n")
    else:
        print(f"Test Failed \n")

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
    # 执行测试
    run_kelvin_operator("samples/data/lena.jpg")

if __name__ == "__main__":
    main()