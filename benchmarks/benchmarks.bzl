# Copyright 2024 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Rules to run Kelvin benchmarks"""

load("@kelvin_sw//build_tools/bazel:kelvin.bzl","kelvin_binary", "kelvin_test", "generate_cc_arrays")
load("@matcha//rules:matcha.bzl", "bin_to_c_file", "matcha_extflash_tar", "sec_flash_binary", "smc_flash_binary", "device_deps")

def kelvin_benchmark_simulator(
        name,
        model,
        iterations,
        test_data = None,
        profile = False,
        kelvin_binary_info = None,
        benchmark_path = "benchmarks",
        hw_test_size = "medium",
        hw_test_tags = [],
        iss_test_size = "small",
        iss_test_tags = [],
        **kwargs):

        if kelvin_binary_info:
            kelvin_test(
                name = kelvin_binary_info['name'],
                srcs = kelvin_binary_info['srcs'],
                hdrs = kelvin_binary_info['hdrs'],
                copts = kelvin_binary_info['copts'],
                deps = kelvin_binary_info['deps'],
                hw_test_size = "medium",
                iss_test_size = "medium",
            )
        else:
            kelvin_headers = ["@kelvin_sw//benchmarks:benchmark.h"]
            model_header_name = "{}_model".format(name)
            bin_to_c_file(
                name = model_header_name,
                srcs = [model],
                var_name = "g_benchmark_model_data",
            )
            kelvin_headers.append(model_header_name)

            if test_data:
                test_data_header_name = "{}_test_data".format(name)
                bin_to_c_file(
                    name = test_data_header_name,
                    srcs = [test_data],
                    var_name = "g_benchmark_test_data",
                )
                kelvin_headers.append(test_data_header_name)

            # Test to run in simulator and MPACT.
            kelvin_test(
                name = "{}".format(name),
                srcs = ["@kelvin_sw//benchmarks:benchmark_kelvin.cc"],
                hdrs = kelvin_headers,
                copts = [
                    "-DITERATIONS={}".format(iterations),
                    "-DBENCHMARK_NAME={}".format(name),
                    "-DTEST_DATA={}".format(1 if test_data else 0),
                    "-DPROFILE={}".format(1 if profile else 0),
                    "-DBENCHMARK_PATH={}".format(benchmark_path),
                ],
                deps = [
                    "@kelvin_sw//crt",
                    "@kelvin_sw//benchmarks:benchmark_header",
                    "@kelvin_sw//benchmarks:cycle_count",
                    "@tflite-micro//tensorflow/lite/micro:micro_framework",
                    "@tflite-micro//tensorflow/lite/micro:system_setup",
                ],
                hw_test_size = hw_test_size,
                hw_test_tags = hw_test_tags,
                iss_test_size = iss_test_size,
                iss_test_tags = iss_test_tags,
            )

def kelvin_benchmark_fpga(
        name,
        model,
        iterations,
        test_data = None,
        profile = False,
        kelvin_binary_info = None,
        benchmark_path = "benchmarks",
        **kwargs):
        _kelvin_benchmark_device(
            name = name,
            model = model,
            device_type = "fpga_nexus",
            iterations = iterations,
            test_data = test_data,
            profile = profile,
            kelvin_binary_info = kelvin_binary_info,
            benchmark_path = benchmark_path,
            **kwargs,
        )

def kelvin_benchmark_asic(
        name,
        model,
        iterations,
        test_data = None,
        profile = False,
        kelvin_binary_info = None,
        benchmark_path = "benchmarks",
        **kwargs):

        _kelvin_benchmark_device(
            name = name,
            model = model,
            device_type = "asic",
            iterations = iterations,
            test_data = test_data,
            profile = profile,
            kelvin_binary_info = kelvin_binary_info,
            benchmark_path = benchmark_path,
            **kwargs,
        )

def kelvin_benchmark_devices(
        name,
        model,
        iterations,
        test_data = None,
        profile = False,
        kelvin_binary_info = None,
        benchmark_path = "benchmarks",
        **kwargs):

        kelvin_benchmark_asic(
            name = "{}_asic".format(name),
            model = model,
            iterations = iterations,
            test_data = test_data,
            profile = profile,
            kelvin_binary_info = kelvin_binary_info,
            benchmark_path = benchmark_path,
            **kwargs,
        )

        kelvin_benchmark_fpga(
            name = "{}_fpga".format(name),
            model = model,
            iterations = iterations,
            test_data = test_data,
            profile = profile,
            kelvin_binary_info = kelvin_binary_info,
            benchmark_path = benchmark_path,
            **kwargs,
        )

        # Create a filegroup to allow building all devices
        native.filegroup(
            name = "{}".format(name),
            srcs = [
                ":{}_asic".format(name),
                ":{}_fpga".format(name),
            ],
            output_group = "device_files",
        )

def _kelvin_benchmark_device(
        name,
        model,
        device_type,
        iterations,
        test_data = None,
        profile = False,
        kelvin_binary_info = None,
        benchmark_path = "benchmarks",
        **kwargs):

        # Creation of binaries for running on FPGA
        smc_flash_binary(
            name = "{}_smc".format(name),
            srcs = [
                "@kelvin_sw//benchmarks:benchmark_smc.c",
                "@kelvin_sw//benchmarks:benchmark.h",
            ],
            copts = ["-DBENCHMARK_NAME={}".format(name)],
            per_device_deps = {
                device_type: device_deps("smc").get(device_type),
            },
            deps = [
                "@matcha//sw/device/lib/dif:ml_top",
                "@matcha//sw/device/tests:test_lib_smc",
                "@matcha//sw/device/lib/dif:i2s",
                "@matcha//sw/device/lib/dif:tlul_mailbox",
                "@kelvin_sw//benchmarks:benchmark_header",
                "@kelvin_sw//benchmarks:cycle_count",
                "@lowrisc_opentitan//sw/device/lib/dif:rv_plic",
                "@lowrisc_opentitan//sw/device/lib/dif:rv_timer",
            ],
        )

        bin_to_c_file(
            name = "{}-smc_bin".format(name),
            srcs = ["{}_smc_{}_bin".format(name, device_type)],
            var_name = "smc_bin",
        )

        sec_flash_binary(
            name = "{}_sec".format(name),
            srcs = [
                "@kelvin_sw//benchmarks:benchmark_sec.c",
                "{}-smc_bin.h".format(name),
                "@kelvin_sw//benchmarks:benchmark.h",
            ],
            copts = [
                "-DBENCHMARK_NAME={}".format(name),
                "-DBENCHMARK_PATH={}".format(benchmark_path),
            ],
            per_device_deps = {
                device_type: device_deps("secure_core").get(device_type),
            },
            deps = [
                "@matcha//sw/device/lib:spi_flash",
                "@matcha//sw/device/tests:test_lib",
                "@matcha//sw/device/lib/dif:smc_ctrl",
                "@matcha//sw/device/lib/dif:tlul_mailbox",
                "@kelvin_sw//benchmarks:benchmark_header",
                "@kelvin_sw//benchmarks:cycle_count",
                "@lowrisc_opentitan//sw/device/lib/dif:rv_plic",
            ],
        )

        # If provided Kelvin binary info, use that instead of the standard
        if kelvin_binary_info:
            kelvin_binary(
                name = "{}_kelvin".format(name),
                srcs = kelvin_binary_info['srcs'],
                copts = kelvin_binary_info['copts'],
                hdrs = kelvin_binary_info['hdrs'],
                deps = kelvin_binary_info['deps'],
            )
        else:
            kelvin_headers = ["@kelvin_sw//benchmarks:benchmark.h"]
            model_header_name = "{}_model".format(name)
            bin_to_c_file(
                name = "{}_model".format(name),
                srcs = [model],
                var_name = "g_benchmark_model_data",
            )
            kelvin_headers.append(model_header_name)

            if test_data:
                test_data_header_name = "{}_test_data".format(name)
                bin_to_c_file(
                    name = test_data_header_name,
                    srcs = [test_data],
                    var_name = "g_benchmark_test_data",
                )
                kelvin_headers.append(test_data_header_name)

            kelvin_binary(
                name = "{}_kelvin".format(name),
                srcs = [
                    "@kelvin_sw//benchmarks:benchmark_kelvin.cc",
                ],
                copts = [
                    "-DITERATIONS={}".format(iterations),
                    "-DBENCHMARK_NAME={}".format(name),
                    "-DTEST_DATA={}".format(1 if test_data else 0),
                    "-DPROFILE={}".format(1 if profile else 0),
                    "-DBENCHMARK_PATH={}".format(benchmark_path),
                ],
                hdrs = kelvin_headers,
                deps = [
                    "@kelvin_sw//benchmarks:benchmark_header",
                    "@kelvin_sw//benchmarks:cycle_count",
                    "@tflite-micro//tensorflow/lite/micro:micro_framework",
                    "@tflite-micro//tensorflow/lite/micro:system_setup",
                ],
            )

        matcha_extflash_tar(
            name = "{}_extflash".format(name),
            kelvin_binary = ":{}_kelvin.bin".format(name),
            sc_binary = ":{}_sec_{}_bin".format(name, device_type),
        )

        # Create a filegroup with all device targets.
        native.filegroup(
            name = "{}".format(name),
            srcs = [
                ":{}_sec".format(name),
                ":{}-smc_bin".format(name),
                ":{}_kelvin".format(name),
                ":{}_extflash".format(name),
            ],
            output_group = "{}_files".format(device_type),
        )
