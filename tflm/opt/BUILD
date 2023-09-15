package(default_visibility = ["//visibility:public"])

cc_library(
    name = "opt",
    srcs = [
        "elementwise_add_s16.cc",
        "elementwise_add_s32.cc",
        "elementwise_add_s8.cc",
        "leaky_relu_s16.cc",
        "leaky_relu_s8.cc",
        "memcpy.cc",
    ],
    hdrs = [
        "opt.h",
        "util.h",
    ],
    target_compatible_with = ["@kelvin_sw//platforms/cpu:kelvin"],
    deps = [
        "//crt:crt",
    ],
    alwayslink = True,
)
