# Copyright 2023 Google LLC
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

load("//build_tools/bazel:kelvin.bzl", "kelvin_test")

cc_library(
    name = "kelvin_test",
    srcs = [
        "kelvin_test.h",
    ],
    deps = [
        "//crt",
    ],
)

kelvin_test(
    name = "perf_counters",
    srcs = [
        "perf_counters.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    copts = select({
        "//:link_tcm_config": ["-DLINK_TCM"],
        "//conditions:default": [],
    }),
)

kelvin_test(
    name = "ecall",
    srcs = [
        "ecall.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
)

kelvin_test(
    name = "mret_fault",
    srcs = [
        "mret_fault.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    iss = False,
    expect_success = False,
    tags = ["intentional_failure"],
)

kelvin_test(
    name = "instr_fault",
    srcs = [
        "instr_fault.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    iss = False,
    expect_success = False,
    tags = ["intentional_failure"],
)

kelvin_test(
    name = "illegal",
    srcs = [
        "illegal.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    iss = False,
    expect_success = False,
    tags = ["intentional_failure"],
)

kelvin_test(
    name = "instr_align_0",
    srcs = [
        "instr_align_0.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    iss = False,
    expect_success = False,
    tags = ["intentional_failure"],
)

kelvin_test(
    name = "instr_align_1",
    srcs = [
        "instr_align_1.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    iss = False,
    expect_success = False,
    tags = ["intentional_failure"],
)

kelvin_test(
    name = "instr_align_2",
    srcs = [
        "instr_align_2.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    iss = False,
    expect_success = False,
    tags = ["intentional_failure"],
)

kelvin_test(
    name = "load_fault_0",
    srcs = [
        "load_fault_0.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    iss = False,
    expect_success = False,
    tags = ["intentional_failure"],
)

kelvin_test(
    name = "load_fault_1",
    srcs = [
        "load_fault_1.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    iss = False,
    expect_success = False,
    tags = ["intentional_failure"],
)

kelvin_test(
    name = "store_fault_0",
    srcs = [
        "store_fault_0.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    iss = False,
    # This passes on core_sim, but should fail on core_mini_sim.
    # expect_success = False,
)

kelvin_test(
    name = "store_fault_1",
    srcs = [
        "store_fault_1.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    iss = False,
    expect_success = False,
    tags = ["intentional_failure"],
)

kelvin_test(
    name = "acset",
    srcs = [
        "acset.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "actr",
    srcs = [
        "actr.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "getvl_test",
    srcs = [
        "getvl_test.cc",
    ],
    hdrs = [
        "kelvin_test.h",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vacc",
    srcs = [
        "vacc.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vadd",
    srcs = [
        "vadd.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "valu",
    srcs = [
        "valu.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vcmp",
    srcs = [
        "vcmp.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vevn_vodd",
    srcs = [
        "vevn_vodd.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

cc_library(
    name = "vdwconv_data",
    srcs = [
        "vdwconv_data.h",
    ],
)

kelvin_test(
    name = "vdwconv",
    srcs = [
        "vdwconv.cc",
    ],
    deps = [
        ":kelvin_test",
        ":vdwconv_data",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vld",
    srcs = [
        "vld.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vmax_vmin",
    srcs = [
        "vmax_vmin.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vmul",
    srcs = [
        "vmul.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vmv",
    srcs = [
        "vmv.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vpadd",
    srcs = [
        "vpadd.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vsel",
    srcs = [
        "vsel.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vshift",
    srcs = [
        "vshift.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vslide",
    srcs = [
        "vslide.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vsrn",
    srcs = [
        "vsrn.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vst",
    srcs = [
        "vst.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vstq",
    srcs = [
        "vstq.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "vzip",
    srcs = [
        "vzip.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
    tags = ["tcm-incompatible"],
)
