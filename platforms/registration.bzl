"""Kelvin toolchain registration."""

def kelvin_register_toolchain(name = "kelvin"):
    native.register_execution_platforms("//platforms/riscv32:kelvin")
    native.register_toolchains("//toolchains/kelvin:all")
