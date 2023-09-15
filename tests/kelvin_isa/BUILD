load("//build_tools/bazel:kelvin.bzl", "kelvin_test")

cc_library(
    name = "kelvin_test",
    srcs = [
        "kelvin_test.h",
    ],
    deps = [
        "//crt:crt",
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
