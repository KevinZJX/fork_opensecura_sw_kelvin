import struct
import time
import numpy as np
from pyrenode3.wrappers import Emulation, Monitor
from System import Array, Byte

class KelvinTester:
    def __init__(self, bin_path, resc_path):
        self.m = Monitor()
        self.e = Emulation()
        self.bin_path = bin_path
        self.m.Parse(f"$bin=@{bin_path}")
        if not self.m.Parse(f"i @{resc_path}"):
            print("Failed to load kelvin.resc!")
            exit()
        self.mach = self.e.get_mach("kelvin")
        self.sysbus = self.mach.sysbus

        # 内存地址定义
        self.ADDR_IN_DATA = 0x5A800000
        self.ADDR_OUT_DATA = 0x5A400000
        self.ADDR_IN_HDR = 0x5AFFFF80
        self.ADDR_OUT_HDR = 0x5AFFFFC0

    def reset(self):
        """复位整个仿真器状态"""
        self.mach.Reset()
        self.m.Parse(f"$bin=@{self.bin_path}")
        print("Simulator Reset Done.\n")

    def run_kernel(self, opcode, input_bytes, width, height, channels=1, params=[0]*10, timeout=5, auto_reset=True):
        if auto_reset:
            self.reset()

        dotnet_data = Array[Byte](input_bytes)
        self.sysbus.WriteBytes(self.ADDR_IN_DATA, dotnet_data, 0, dotnet_data.Length)

        header_data = struct.pack("6I10I", opcode, width, height, channels, width, 0, *params)
        self.sysbus.WriteBytes(self.ADDR_IN_HDR, bytearray(header_data), 0, len(header_data))

        self.e.StartAll()
        self.sysbus.ml_top_controlblock.WriteDoubleWord(0xc, 0)

        start_time = time.time()
        while not self.sysbus.cpu2.IsHalted:
            time.sleep(0.01)
            if time.time() - start_time > 5:
                print("Error: Simulation Timeout!\n")
                break


        status = self.sysbus.ReadDoubleWord(self.ADDR_OUT_HDR)
        out_len = self.sysbus.ReadDoubleWord(self.ADDR_OUT_HDR + 4)
        # cycles = self.sysbus.ReadDoubleWord(self.ADDR_OUT_HDR + 8)
        cycles = self.sysbus.cpu2.ExecutedInstructions

        result_bytes = self.sysbus.ReadBytes(self.ADDR_OUT_DATA, out_len)
        return status, out_len, bytes(result_bytes), cycles


