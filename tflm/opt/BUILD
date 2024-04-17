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

cc_library(
    name = "opt",
    srcs = [
        "conv.cc",
        "depthwise_conv_s16.cc",
        "depthwise_conv_s8.cc",
        "elementwise_add_s16.cc",
        "elementwise_add_s32.cc",
        "elementwise_add_s8.cc",
        "leaky_relu_s16.cc",
        "leaky_relu_s8.cc",
        "memcpy.cc",
        "max_pool_s8.cc"
    ],
    hdrs = [
        "opt.h",
        "util.h",
    ],
    target_compatible_with = ["@kelvin_sw//platforms/cpu:kelvin"],
    deps = [
        "//crt",
        "@tflite-micro//tensorflow/lite/kernels/internal:common",
        "@tflite-micro//tensorflow/lite/kernels/internal:reference_base",
    ],
    alwayslink = True,
)
