workspace(name = "kelvin")

load("@bazel_tools//tools/build_defs/repo:utils.bzl", "maybe")
load("@bazel_tools//tools/build_defs/repo:http.bzl", "http_archive")

# Kelvin toolchain
new_local_repository(
    name = "kelvin-gcc",
    build_file = "third_party/kelvin-gcc/BUILD",
    path = "../../cache/toolchain_kelvin",
)

# CRT is the Compiler Repository Toolkit.  It contains the configuration for
# the windows compiler.
maybe(
    http_archive,
    name = "crt",
    url = "https://github.com/lowRISC/crt/archive/refs/tags/v0.3.4.tar.gz",
    sha256 = "01a66778d1a0d5bbfb4ba30e72bd6876d0c20766d0b1921ab36ca3350cb48c60",
    strip_prefix = "crt-0.3.4",
)

load("@crt//:repos.bzl", "crt_repos")

crt_repos()

load("@crt//:deps.bzl", "crt_deps")

crt_deps()

load("//platforms:registration.bzl", "kelvin_register_toolchain")

kelvin_register_toolchain()
