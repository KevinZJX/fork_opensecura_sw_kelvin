"""Kelvin dependency repository setup."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "new_git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def kelvin_repos():
    """"Setup Kelvin dependency repositories."""

    # Kelvin toolchain
    native.new_local_repository(
        name = "kelvin-gcc",
        build_file = "third_party/kelvin-gcc/BUILD",
        path = "../../cache/toolchain_kelvin",
    )

    # CRT is the Compiler Repository Toolkit.  It contains the configuration for
    # the windows compiler.
    maybe(
        http_archive,
        name = "crt",
        url = "https://github.com/lowRISC/crt/archive/refs/tags/v0.3.4.tar.gz",
        sha256 = "01a66778d1a0d5bbfb4ba30e72bd6876d0c20766d0b1921ab36ca3350cb48c60",
        strip_prefix = "crt-0.3.4",
    )

    # risc-v isa test
    new_git_repository(
        name = "riscv-tests",
        commit = "d649367a1386609da3d10e9e6d388f98781dd35f",
        build_file = "//third_party/riscv:BUILD.riscv-tests",
        shallow_since = "1636745372 -0800",
        remote = "https://spacebeaker.googlesource.com/shodan/3p/riscv/riscv-tests",
    )
