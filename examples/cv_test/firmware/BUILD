
load("//build_tools/bazel:kelvin.bzl", "kelvin_binary")

kelvin_binary(
    name = "kelvin_firmware",
    srcs = [
        "src/main.cpp",
        "src/cvtColor.cpp",
    ],
    hdrs = [
        "include/common.h",
        "include/memory_layout.h",
        "include/operators.h",
    ],
    deps = [
        "//crt",
    ],
)
