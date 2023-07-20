# Build riscv-tests based on Kelvin linker script and termination condition.

load("//build_tools/bazel:kelvin.bzl", "kelvin_binary")

cc_library(
    name = "riscv_tests_base",
    srcs = [
        "test_main.cc",
    ],
    hdrs = [
        "riscv_test.h",
        "@riscv-tests//:isa/macros/scalar/test_macros.h",
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
    "fence_i",
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

[kelvin_binary(
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
    is_riscv_test = True,
    deps = [
        ":riscv_tests_base",
    ],
) for test in RV32UI_TESTS]

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

[kelvin_binary(
    name = "rv32um_{}".format(test),
    srcs = [
        "@riscv-tests//:isa/rv32um/{}.S".format(test),
    ],
    copts = [
        "-Itests/riscv-tests",
        "-Iexternal/riscv-tests/isa/macros/scalar",
        "-Wno-variadic-macros",
    ],
    is_riscv_test = True,
    deps = [
        ":riscv_tests_base",
    ],
) for test in RV32UM_TESTS]

kelvin_binary(
    name = "branch_modulo_test",
    srcs = [
        "branch_modulo_test.cc",
    ],
    deps = [
        "//crt:crt_header",
    ],
)

kelvin_binary(
    name = "branch_div_test",
    srcs = [
        "branch_div_test.cc",
    ],
    deps = [
        "//crt:crt_header",
    ],
)
