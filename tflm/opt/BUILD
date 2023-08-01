package(default_visibility = ["//visibility:public"])

cc_library(
    name = "opt",
    srcs = [
        "memcpy.cc",
    ],
    hdrs = [
        "opt.h",
    ],
    deps = [
        "//crt:crt_header",
    ],
    alwayslink = True,
    target_compatible_with = ["@kelvin_sw//platforms/cpu:kelvin"],
)
