# Build riscv-tests based on Kelvin linker script and termination condition.

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

kelvin_test(
    name = "branch_modulo_test",
    srcs = [
        "branch_modulo_test.cc",
    ],
    hw_test_size = "small",
    deps = [
        "//crt",
    ],
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
)
