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

# Build riscv-tests based on Kelvin linker script and termination condition.

package(default_visibility = ["//visibility:public"])
load("//build_tools/bazel:kelvin.bzl", "kelvin_test")

cc_library(
    name = "riscv_tests_base",
    srcs = [
        "test_main.cc",
    ],
    hdrs = [
        "riscv_test.h",
        "@riscv-tests//:isa/macros/scalar/test_macros.h",
    ],
    target_compatible_with = ["@kelvin_sw//platforms/cpu:kelvin"],
    deps = [
        # only include the header since riscv tests have their own start.S.
        "//crt:crt_gloss_only",
    ],
)

RV32UI_TESTS = [
    "add",
    "addi",
    "and",
    "andi",
    "auipc",
    "beq",
    "bge",
    "bgeu",
    "blt",
    "bltu",
    "bne",
    "jalr",
    "jal",
    "lb",
    "lbu",
    "lh",
    "lhu",
    "lui",
    "lw",
    "ori",
    "or",
    "sb",
    "sh",
    "simple",
    "slli",
    "sll",
    "slti",
    "sltiu",
    "slt",
    "sltu",
    "srai",
    "sra",
    "srli",
    "srl",
    "sub",
    "sw",
    "xori",
    "xor",
]

[kelvin_test(
    name = "rv32ui_{}".format(test),
    srcs = [
        # riscv-tests use the rv64 code to set up rv32 tests.
        "@riscv-tests//:isa/rv64ui/{}.S".format(test),
    ],
    # The include paths are set explicitly because riscv-tests has special
    # include dependencies.
    copts = [
        "-Itests/riscv-tests",
        "-Iexternal/riscv-tests/isa/macros/scalar",
        "-Wno-variadic-macros",
    ],
    defines = [
        "RVTEST_RV64U=RVTEST_RV32U",
    ],
    hw_test_size = "small",
    deps = [
        ":riscv_tests_base",
    ],
) for test in RV32UI_TESTS]

kelvin_test(
    name = "rv32ui_fence_i",
    srcs = [
        "@riscv-tests//:isa/rv64ui/fence_i.S",
    ],
    # The include paths are set explicitly because riscv-tests has special
    # include dependencies.
    copts = [
        "-Itests/riscv-tests",
        "-Iexternal/riscv-tests/isa/macros/scalar",
        "-Wno-variadic-macros",
    ],
    defines = [
        "RVTEST_RV64U=RVTEST_RV32U",
    ],
    hw_test_size = "small",
    deps = [
        ":riscv_tests_base",
    ],
    tags = ["tcm-incompatible"],
)

RV32UM_TESTS = [
    "div",
    "divu",
    "mul",
    "mulh",
    "mulhsu",
    "mulhu",
    "rem",
    "remu",
]

[kelvin_test(
    name = "rv32um_{}".format(test),
    srcs = [
        "@riscv-tests//:isa/rv32um/{}.S".format(test),
    ],
    copts = [
        "-Itests/riscv-tests",
        "-Iexternal/riscv-tests/isa/macros/scalar",
        "-Wno-variadic-macros",
    ],
    hw_test_size = "small",
    deps = [
        ":riscv_tests_base",
    ],
) for test in RV32UM_TESTS]

RV32UZBB_TESTS_64 = [
    "andn",
    "max",
    "maxu",
    "min",
    "minu",
    "orn",
    "sext_b",
    "sext_h",
    "xnor",
    "zext_h",
]

[kelvin_test(
    name = "rv32uzbb_{}".format(test),
    srcs = [
        "@riscv-tests//:isa/rv64uzbb/{}.S".format(test),
    ],
    copts = [
        "-Itests/riscv-tests",
        "-Iexternal/riscv-tests/isa/macros/scalar",
        "-Wno-variadic-macros",
    ],
    defines = [
        "RVTEST_RV64U=RVTEST_RV32U"
    ],
    hw_test_size = "small",
    deps = [
        ":riscv_tests_base",
    ],
) for test in RV32UZBB_TESTS_64]

RV32UZBB_TESTS = [
    "clz",
    "cpop",
    "ctz",
    "orc_b",
    "rev8",
    "rol",
    "ror",
    "rori",
]

[kelvin_test(
    name = "rv32uzbb_{}".format(test),
    srcs = [
        "@riscv-tests//:isa/rv32uzbb/{}.S".format(test),
    ],
    copts = [
        "-Itests/riscv-tests",
        "-Iexternal/riscv-tests/isa/macros/scalar",
        "-Wno-variadic-macros",
    ],
    hw_test_size = "small",
    deps = [
        ":riscv_tests_base",
    ],
) for test in RV32UZBB_TESTS]

kelvin_test(
    name = "rv32mi_mcsr",
    srcs = [
        "@riscv-tests//:isa/rv64mi/mcsr.S",
    ],
    copts = [
        "-Itests/riscv-tests",
        "-Iexternal/riscv-tests/isa/macros/scalar",
        "-Wno-variadic-macros",
    ],
    defines = [
        "RVTEST_RV64M=RVTEST_RV32M",
    ],
    hw_test_size = "small",
    deps = [
        ":riscv_tests_base",
    ],
)

# kelvin_test(
#     name = "rv32mi_illegal",
#     srcs = [
#         "@riscv-tests//:isa/rv64mi/illegal.S",
#     ],
#     copts = [
#         "-Itests/riscv-tests",
#         "-Iexternal/riscv-tests/isa/macros/scalar",
#         "-Wno-variadic-macros",
#     ],
#     defines = [
#         "RVTEST_RV64M=RVTEST_RV32M",
#     ],
#     hw_test_size = "small",
#     deps = [
#         ":riscv_tests_base",
#     ],
# )

kelvin_test(
    name = "rv32mi_csr",
    srcs = [
        "@riscv-tests//:isa/rv64si/csr.S",
    ],
    copts = [
        "-Itests/riscv-tests",
        "-Iexternal/riscv-tests/isa/macros/scalar",
        "-Wno-variadic-macros",
    ],
    defines = [
        "RVTEST_RV64S=RVTEST_RV32M",
        "__MACHINE_MODE",
    ],
    hw_test_size = "small",
    deps = [
        ":riscv_tests_base",
    ],
)

cc_library(
    name = "dhrystone_lib",
    srcs = [
        "@riscv-tests//:benchmarks/dhrystone/dhrystone.c",
    ],
    hdrs = [
        "@riscv-tests//:benchmarks/dhrystone/dhrystone.h",
    ],
    copts = [
        "-Wno-error",
        "-Wno-implicit-int",
        "-Wno-implicit-function-declaration",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "rv_dhrystone",
    srcs = [
        "@riscv-tests//:benchmarks/dhrystone/dhrystone_main.c",
    ],
    hdrs =[
        "@riscv-tests//:benchmarks/common/util.h",
        "@riscv-tests//:env/encoding.h",
    ],
    deps = [
        ":dhrystone_lib",
        "//crt",
        "//benchmarks:benchmark_header",
    ],
    copts = [
        "-Iexternal/riscv-tests/benchmarks/common",
        "-Iexternal/riscv-tests/env",
        "-Wno-error",
        "-Wno-implicit-int",
        "-Wno-implicit-function-declaration",
    ],
    hw_test_size = "large",
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "alu",
    srcs = [
        "alu.cc",
    ],
    hw_test_size = "small",
    deps = [
        "//crt",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "branch_modulo_test",
    srcs = [
        "branch_modulo_test.cc",
    ],
    hw_test_size = "small",
    deps = [
        "//crt",
    ],
    tags = ["tcm-incompatible"],
)

kelvin_test(
    name = "branch_div_test",
    srcs = [
        "branch_div_test.cc",
    ],
    hw_test_size = "small",
    deps = [
        "//crt",
    ],
    tags = ["tcm-incompatible"],
)
