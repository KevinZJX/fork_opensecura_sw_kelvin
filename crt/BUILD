package(default_visibility = ["//visibility:public"])

filegroup(
    name = "kelvin_linker",
    srcs = ["kelvin.ld"],
)

cc_library(
    name = "crt",
    srcs = [
        "crt.S",
        "kelvin_gloss.cc",
        "kelvin_start.S",
    ],
    target_compatible_with = ["@kelvin_sw//platforms/cpu:kelvin"],
    deps = [
        ":crt_header",
    ],
)

# Used by binaries with their own starting asm.
cc_library(
    name = "crt_gloss_only",
    srcs = [
        "kelvin_gloss.cc",
    ],
    target_compatible_with = ["@kelvin_sw//platforms/cpu:kelvin"],
    deps = [
        ":crt_header",
    ],
)

cc_library(
    name = "crt_header",
    hdrs = [
        "kelvin.h",
        "kelvin_intrinsics.h",
        "printf_traits.h",
    ],
)
