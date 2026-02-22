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

load("//build_tools/bazel:kelvin.bzl", "kelvin_binary")

kelvin_binary(
    name = "cv_test",
    srcs = [
        "cv_test.c",
    ],
    hdrs = [
        "memory_layout.h",
    ],
    deps = [
        "//crt",
    ],
)


kelvin_binary(
    name = "yu12toyv12",
    srcs = [
        "yu12toyv12.c",
    ],
    hdrs = [
        "memory_layout.h",
    ],
    deps = [
        "//crt",
    ],
)
