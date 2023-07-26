load("//build_tools/bazel:kelvin.bzl", "kelvin_binary", "generate_cc_arrays")
package(default_visibility = ["//visibility:public"])

kelvin_binary(
    name = "person_detection",
    srcs = [
        "person_detection.cc",
        "person_detect_tflite.cc",
        "person_bmp.cc",
    ],
    hdrs = [
        "person_bmp.h",
        "person_detect_tflite.h",
    ],
    deps = [
        "//crt:crt_header",
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
