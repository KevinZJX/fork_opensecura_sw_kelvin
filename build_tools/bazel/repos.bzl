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
    http_archive(
        name = "riscv-tests",
        build_file = "//third_party/riscv:BUILD.riscv-tests",
        sha256 = "1c7eb58edd7399b3ad2f9624a2003862cd87a6904237a737f39cd3978bab46a8",
        urls = ["https://github.com/riscv-software-src/riscv-tests/archive/d4eaa5bd6674b51d3b9b24913713c4638e99cdd9.tar.gz"],
        strip_prefix = "riscv-tests-d4eaa5bd6674b51d3b9b24913713c4638e99cdd9",
    )
