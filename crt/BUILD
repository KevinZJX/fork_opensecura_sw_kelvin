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

package(default_visibility = ["//visibility:public"])

filegroup(
    name = "kelvin_linker",
    srcs = ["kelvin.ld"],
)

cc_library(
    name = "crt",
    srcs = [
        "crt.S",
        "kelvin_gloss.cc",
        "kelvin_start.S",
    ],
    target_compatible_with = ["@kelvin_sw//platforms/cpu:kelvin"],
    deps = [
        ":crt_header",
    ],
)

# Used by binaries with their own starting asm.
cc_library(
    name = "crt_gloss_only",
    srcs = [
        "kelvin_gloss.cc",
    ],
    target_compatible_with = ["@kelvin_sw//platforms/cpu:kelvin"],
    deps = [
        ":crt_header",
    ],
)

cc_library(
    name = "crt_header",
    hdrs = [
        "kelvin.h",
        "kelvin_intrinsics.h",
        "printf_traits.h",
    ],
)
