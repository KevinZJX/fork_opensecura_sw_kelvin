package(default_visibility = ["//visibility:public"])

cc_library(
    name = "opt",
    srcs = [
        "elementwise_add_s8.cc",
        "elementwise_add_s16.cc",
        "elementwise_add_s32.cc",
        "memcpy.cc",
    ],
    hdrs = [
        "opt.h",
        "util.h",
    ],
    deps = [
        "//crt:crt_header",
    ],
    alwayslink = True,
    target_compatible_with = ["@kelvin_sw//platforms/cpu:kelvin"],
)
