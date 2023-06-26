load("//build_tools/bazel:kelvin.bzl", "kelvin_binary")

kelvin_binary(
  name = "getvl_test",
  srcs = [
    "getvl_test.cc",
  ],
  hdrs = [
    "kelvin_test.h",
  ],
  copts = [
    "-Wno-address",
  ],
  deps = [
    "//crt:crt_header",
  ]
)
