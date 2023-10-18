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

load("//build_tools/bazel:kelvin.bzl", "generate_cc_arrays", "kelvin_binary")

package(default_visibility = ["//visibility:public"])

kelvin_binary(
    name = "person_detection",
    srcs = [
        "person_bmp.cc",
        "person_detect_tflite.cc",
        "person_detection.cc",
    ],
    hdrs = [
        "person_bmp.h",
        "person_detect_tflite.h",
    ],
    deps = [
        "//crt",
        "@tflite-micro//tensorflow/lite/micro:micro_framework",
        "@tflite-micro//tensorflow/lite/micro:system_setup",
    ],
)

generate_cc_arrays(
    name = "person_bmp_cc",
    src = "@tflite-micro//tensorflow/lite/micro/examples/person_detection:testdata/person.bmp",
    out = "person_bmp.cc",
)

generate_cc_arrays(
    name = "person_bmp_h",
    src = "@tflite-micro//tensorflow/lite/micro/examples/person_detection:testdata/person.bmp",
    out = "person_bmp.h",
)

generate_cc_arrays(
    name = "person_detect_tflite_cc",
    src = "@tflite-micro//tensorflow/lite/micro/models:person_detect.tflite",
    out = "person_detect_tflite.cc",
)

generate_cc_arrays(
    name = "person_detect_tflite_h",
    src = "@tflite-micro//tensorflow/lite/micro/models:person_detect.tflite",
    out = "person_detect_tflite.h",
)
