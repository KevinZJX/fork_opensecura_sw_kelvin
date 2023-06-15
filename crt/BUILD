package(default_visibility = ["//visibility:public"])

# Exports the kelvin files to be used by the kelvin_binary()
exports_files([
    "crt.S",
    "kelvin.ld",
    "kelvin_gloss.cc",
    "kelvin_start.S",
])

cc_library(
    name = "crt_header",
    hdrs = [
        "kelvin.h",
        "kelvin_intrinsics.h",
    ],
)
