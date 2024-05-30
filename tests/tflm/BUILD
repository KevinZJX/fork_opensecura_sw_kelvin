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

package(default_visibility = ["//visibility:public"])

kelvin_test(
    name = "add_test",
    srcs = [
        "@tflite-micro//tensorflow/lite/micro/kernels:add_test.cc",
    ],
    hw_test_size = "large",
    deps = [
        "//crt",
        "@tflite-micro//tensorflow/lite/c:common",
        "@tflite-micro//tensorflow/lite/kernels/internal:tensor",
        "@tflite-micro//tensorflow/lite/micro:micro_utils",
        "@tflite-micro//tensorflow/lite/micro:test_helpers",
        "@tflite-micro//tensorflow/lite/micro/kernels:kernel_runner",
        "@tflite-micro//tensorflow/lite/micro/testing:micro_test",
    ],
)

kelvin_test(
    name = "conv_test",
    srcs = [
        "@tflite-micro//tensorflow/lite/micro/kernels:conv_test.cc",
    ],
    hw_test_size = "enormous",
    deps = [
        "//crt:crt_header",
        "@tflite-micro//tensorflow/lite/c:common",
        "@tflite-micro//tensorflow/lite/kernels/internal:tensor",
        "@tflite-micro//tensorflow/lite/micro:micro_utils",
        "@tflite-micro//tensorflow/lite/micro:test_helpers",
        "@tflite-micro//tensorflow/lite/micro/kernels:conv_test_common",
        "@tflite-micro//tensorflow/lite/micro/kernels:kernel_runner",
        "@tflite-micro//tensorflow/lite/micro/kernels/testdata:conv_test_data",
        "@tflite-micro//tensorflow/lite/micro/testing:micro_test",
    ],
)

kelvin_test(
    name = "depthwise_conv_test",
    srcs = [
        "@tflite-micro//tensorflow/lite/micro/kernels:depthwise_conv_test.cc",
    ],
    hw_test_size = "enormous",
    deps = [
        "//crt:crt_header",
        "@tflite-micro//tensorflow/lite/c:common",
        "@tflite-micro//tensorflow/lite/kernels/internal:tensor",
        "@tflite-micro//tensorflow/lite/micro:micro_utils",
        "@tflite-micro//tensorflow/lite/micro:test_helpers",
        "@tflite-micro//tensorflow/lite/micro/kernels:kernel_runner",
        "@tflite-micro//tensorflow/lite/micro/testing:micro_test",
    ],
)

kelvin_test(
    name = "leaky_relu_test",
    srcs = [
        "@tflite-micro//tensorflow/lite/micro/kernels:leaky_relu_test.cc",
    ],
    deps = [
        "//crt",
        "@tflite-micro//tensorflow/lite/c:common",
        "@tflite-micro//tensorflow/lite/kernels/internal:tensor",
        "@tflite-micro//tensorflow/lite/micro:micro_utils",
        "@tflite-micro//tensorflow/lite/micro:test_helpers",
        "@tflite-micro//tensorflow/lite/micro/kernels:kernel_runner",
        "@tflite-micro//tensorflow/lite/micro/testing:micro_test",
    ],
)

kelvin_test(
    name = "logistic_test",
    srcs = [
        "@tflite-micro//tensorflow/lite/micro/kernels:logistic_test.cc",
    ],
    deps = [
        "//crt",
        "@tflite-micro//tensorflow/lite/c:common",
        "@tflite-micro//tensorflow/lite/kernels/internal:tensor",
        "@tflite-micro//tensorflow/lite/micro:micro_utils",
        "@tflite-micro//tensorflow/lite/micro:test_helpers",
        "@tflite-micro//tensorflow/lite/micro/kernels:kernel_runner",
        "@tflite-micro//tensorflow/lite/micro/testing:micro_test",
    ],
)

kelvin_test(
    name = "pooling_test",
    srcs = [
        "@tflite-micro//tensorflow/lite/micro/kernels:pooling_test.cc",
    ],
    hw_test_size = "large",
    deps = [
        "//crt",
        "@tflite-micro//tensorflow/lite/c:common",
        "@tflite-micro//tensorflow/lite/kernels/internal:tensor",
        "@tflite-micro//tensorflow/lite/micro:micro_utils",
        "@tflite-micro//tensorflow/lite/micro:test_helpers",
        "@tflite-micro//tensorflow/lite/micro/kernels:kernel_runner",
        "@tflite-micro//tensorflow/lite/micro/testing:micro_test",
    ],
)

kelvin_test(
    name = "reshape_test",
    srcs = [
        "@tflite-micro//tensorflow/lite/micro/kernels:reshape_test.cc",
    ],
    hw_test_size = "large",
    deps = [
        "//crt",
        "@tflite-micro//tensorflow/lite/c:common",
        "@tflite-micro//tensorflow/lite/kernels/internal:tensor",
        "@tflite-micro//tensorflow/lite/micro:micro_utils",
        "@tflite-micro//tensorflow/lite/micro:test_helpers",
        "@tflite-micro//tensorflow/lite/micro/kernels:kernel_runner",
        "@tflite-micro//tensorflow/lite/micro/testing:micro_test",
    ],
)
