# Copyright lowRISC contributors.
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0

# Forked from @crt//toolchains/lowrisc_rv32imcb/BUILD.bazel

load("@crt//config:compiler.bzl", "setup")
load("//platforms/riscv32:devices.bzl", "DEVICES")

package(default_visibility = ["//visibility:public"])

SYSTEM_INCLUDE_PATHS = [
    "external/kelvin-gcc/riscv32-unknown-elf/include/c++/10.2.0",
    "external/kelvin-gcc/riscv32-unknown-elf/include/c++/10.2.0/backward",
    "external/kelvin-gcc/riscv32-unknown-elf/include/c++/10.2.0/riscv32-unknown-elf",
    "external/kelvin-gcc/lib/gcc/riscv32-unknown-elf/10.2.0/include",
    "external/kelvin-gcc/riscv32-unknown-elf/include",
    "external/kelvin-gcc/lib/gcc/riscv32-unknown-elf/10.2.0/include-fixed",
]

filegroup(
    name = "compiler_components",
    srcs = [
        "//toolchains/kelvin/wrappers:all",
        "@kelvin-gcc//:all",
    ],
)

[setup(
    name = device.name,
    architecture = device.architecture,
    artifact_naming = device.artifact_naming,
    compiler_components = ":compiler_components",
    constraints = device.constraints,
    feature_set = device.feature_set,
    include_directories = SYSTEM_INCLUDE_PATHS,
    params = {
        "compiler": "gcc",
    },
    substitutions = device.substitutions,
    tools = {
        "ar": "wrappers/ar",
        "cpp": "wrappers/cpp",
        "gcc": "wrappers/gcc",
        "gcov": "wrappers/gcov",
        "ld": "wrappers/ld",
        "nm": "wrappers/nm",
        "objcopy": "wrappers/objcopy",
        "objdump": "wrappers/objdump",
        "strip": "wrappers/strip",
    },
) for device in DEVICES]
