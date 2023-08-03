workspace(name = "kelvin_sw")

load("//build_tools/bazel:repos.bzl", "kelvin_repos")

kelvin_repos()

# Register Kelvin toolchain
load("//platforms:registration.bzl", "kelvin_register_toolchain")

kelvin_register_toolchain()
