# Copyright 2023 Google LLC
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
# Computer vision examples.

load("//build_tools/bazel:kelvin.bzl", "kelvin_test")

cc_library(
    name = "test_helper",
    hdrs = [
        "test_helper.h",
    ],
)

kelvin_test(
    name = "diff_test",
    srcs = [
        "diff.cc",
        "diff_test.cc",
    ],
    hdrs = [
        "diff.h",
    ],
    deps = [
        ":test_helper",
        "//crt:crt_header",
    ],
)

kelvin_test(
    name = "downsample_test",
    srcs = [
        "downsample.cc",
        "downsample_test.cc",
    ],
    hdrs = [
        "downsample.h",
    ],
    hw_test_size = "small",
    deps = [
        ":test_helper",
        "//crt:crt_header",
    ],
)

kelvin_test(
    name = "extrema_test",
    srcs = [
        "extrema.cc",
        "extrema_test.cc",
    ],
    hdrs = [
        "extrema.h",
    ],
    deps = [
        ":test_helper",
        "//crt:crt_header",
    ],
)

kelvin_test(
    name = "gaussian_test",
    srcs = [
        "gaussian.cc",
        "gaussian_test.cc",
    ],
    hdrs = [
        "gaussian.h",
    ],
    deps = [
        ":test_helper",
        "//crt:crt_header",
    ],
)

kelvin_test(
    name = "shift_gaussian_test",
    srcs = [
        "shift_gaussian.cc",
        "shift_gaussian_test.cc",
    ],
    hdrs = [
        "shift_gaussian.h",
    ],
    deps = [
        ":test_helper",
        "//crt:crt_header",
    ],
)

kelvin_test(
    name = "upsample_test",
    srcs = [
        "upsample.cc",
        "upsample_test.cc",
    ],
    hdrs = [
        "upsample.h",
    ],
    hw_test_size = "small",
    deps = [
        ":test_helper",
        "//crt:crt_header",
    ],
)
