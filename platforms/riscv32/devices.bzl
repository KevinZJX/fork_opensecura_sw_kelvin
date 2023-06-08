load("@crt//config:device.bzl", "device_config")

DEVICES = [
    device_config(
        name = "kelvin",
        architecture = "rv32im",
        feature_set = "//platforms/riscv32/features:rv32im",
        constraints = [
            "//platforms/cpu:kelvin",
            "@platforms//os:none",
        ],
        substitutions = {
            "ARCHITECTURE": "rv32im",
            "ABI": "ilp32",
            "CMODEL": "medany",
            "[STACK_PROTECTOR]": "-fstack-protector-strong",
        },
    ),
]
