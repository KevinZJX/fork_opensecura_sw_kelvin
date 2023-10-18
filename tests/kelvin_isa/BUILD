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

load("//build_tools/bazel:kelvin.bzl", "kelvin_test")

cc_library(
    name = "kelvin_test",
    srcs = [
        "kelvin_test.h",
    ],
    deps = [
        "//crt",
    ],
)

kelvin_test(
    name = "getvl_test",
    srcs = [
        "getvl_test.cc",
    ],
    hdrs = [
        "kelvin_test.h",
    ],
    deps = [
        ":kelvin_test",
    ],
)

cc_library(
    name = "vdwconv_data",
    srcs = [
        "vdwconv_data.h",
    ],
)

kelvin_test(
    name = "vdwconv",
    srcs = [
        "vdwconv.cc",
    ],
    deps = [
        ":kelvin_test",
        ":vdwconv_data",
    ],
)

kelvin_test(
    name = "vld",
    srcs = [
        "vld.cc",
    ],
    deps = [
        ":kelvin_test",
    ],
)

kelvin_test(
    name = "vpadd",
    srcs = [
        "vpadd.cc",
    ],
    hw_test_size = "small",
    deps = [
        ":kelvin_test",
    ],
)
