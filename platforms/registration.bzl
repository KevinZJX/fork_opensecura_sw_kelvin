"""Kelvin toolchain registration."""

def kelvin_register_toolchain(name = "kelvin"):
    native.register_execution_platforms("@kelvin_sw//platforms/riscv32:kelvin")
    native.register_toolchains("@kelvin_sw//toolchains/kelvin:all")
