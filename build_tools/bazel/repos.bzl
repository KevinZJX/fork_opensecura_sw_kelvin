"""Kelvin dependency repository setup."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def kelvin_sw_repos():
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

def tflm_repos():
    """Setup Tensorflow Lite For Microcontrollers repositories."""
    # Tensorflow Lite for Microcontrollers
    native.local_repository(
        name = "tflite-micro",
        path = "../../sw/tflite-micro",
    )

    maybe(
        http_archive,
        name = "gemmlowp",
        sha256 = "43146e6f56cb5218a8caaab6b5d1601a083f1f31c06ff474a4378a7d35be9cfb",  # SHARED_GEMMLOWP_SHA
        strip_prefix = "gemmlowp-fda83bdc38b118cc6b56753bd540caa49e570745",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/gemmlowp/archive/fda83bdc38b118cc6b56753bd540caa49e570745.zip",
            "https://github.com/google/gemmlowp/archive/fda83bdc38b118cc6b56753bd540caa49e570745.zip",
        ],
        patches = [
            "@kelvin_sw//third_party/gemmlowp:pthread.patch",
        ],
        patch_args = [
            "-p1",
        ],
    )

    maybe(
        http_archive,
        name = "ruy",
        sha256 = "da5ec0cc07472bdb21589b0b51c8f3d7f75d2ed6230b794912adf213838d289a",
        strip_prefix = "ruy-54774a7a2cf85963777289193629d4bd42de4a59",
        urls = [
            "https://storage.googleapis.com/mirror.tensorflow.org/github.com/google/ruy/archive/54774a7a2cf85963777289193629d4bd42de4a59.zip",
            "https://github.com/google/ruy/archive/54774a7a2cf85963777289193629d4bd42de4a59.zip",
        ],
        build_file = "@tflite-micro//third_party/ruy:BUILD",
        patches = [
            "@kelvin_sw//third_party/ruy:pthread.patch",
        ],
        patch_args = [
            "-p1",
        ],
    )

    maybe(
        http_archive,
        name = "rules_python",
        sha256 = "497ca47374f48c8b067d786b512ac10a276211810f4a580178ee9b9ad139323a",
        strip_prefix = "rules_python-0.16.1",
        url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.16.1.tar.gz",
    )

    maybe(
        http_archive,
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-faf56fb3df11287f26dbc66fdedf60a2fc2c6631",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/faf56fb3df11287f26dbc66fdedf60a2fc2c6631.zip"],
        sha256 = "a185aa68c93b9f62c80fcb3aadc3c83c763854750dc3f38be1dadcb7be223837",
    )

def model_repos():
    native.new_local_repository(
        name = "ml-models",
        path = "../../ml/ml-models",
        build_file = "third_party/ml-models/BUILD",
    )
