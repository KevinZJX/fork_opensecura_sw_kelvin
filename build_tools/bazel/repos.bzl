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

"""Kelvin dependency repository setup."""

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:git.bzl", "git_repository")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

def kelvin_sw_repos():
    """"Setup Kelvin dependency repositories."""

    # Kelvin toolchain
    native.new_local_repository(
        name = "kelvin-gcc",
        build_file = "@kelvin_sw//third_party/kelvin-gcc:BUILD.kelvin-gcc",
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

    #risc-v isa test
    git_repository(
        name = "riscv-tests",
        build_file = "@kelvin_sw//third_party/riscv:BUILD.riscv-tests",
        remote = "https://github.com/riscv-software-src/riscv-tests",
        commit = "d4eaa5bd6674b51d3b9b24913713c4638e99cdd9",
        recursive_init_submodules = True,
        patch_args = [
            "-p1",
        ],
        patches = [
            "@kelvin_sw//tests/riscv-tests:0001-mcsr.patch",
            "@kelvin_sw//tests/riscv-tests:0002-fixes-for-kelvin.patch",
            "@kelvin_sw//tests/riscv-tests:0003-dhrystone-test-on-fpga.patch",
        ],
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
        name = "rules_python",
        sha256 = "9d04041ac92a0985e344235f5d946f71ac543f1b1565f2cdbc9a2aaee8adf55b",
        strip_prefix = "rules_python-0.26.0",
        url = "https://github.com/bazelbuild/rules_python/archive/refs/tags/0.26.0.tar.gz",
    )

    maybe(
        http_archive,
        name = "pybind11_bazel",
        strip_prefix = "pybind11_bazel-faf56fb3df11287f26dbc66fdedf60a2fc2c6631",
        urls = ["https://github.com/pybind/pybind11_bazel/archive/faf56fb3df11287f26dbc66fdedf60a2fc2c6631.zip"],
        sha256 = "a185aa68c93b9f62c80fcb3aadc3c83c763854750dc3f38be1dadcb7be223837",
    )
