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
