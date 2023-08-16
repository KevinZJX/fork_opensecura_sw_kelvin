workspace(name = "kelvin_sw")

load("//build_tools/bazel:repos.bzl", "kelvin_sw_repos", "model_repos", "tflm_repos")
kelvin_sw_repos()

# Register Kelvin toolchain
load("//platforms:registration.bzl", "kelvin_register_toolchain")

kelvin_register_toolchain()

tflm_repos()

load("@tflite-micro//tensorflow:workspace.bzl", "tf_repositories")
tf_repositories()

load("@rules_python//python:pip.bzl", "pip_parse")
pip_parse(
    name = "tflm_pip_deps",
    requirements_lock = "@tflite-micro//third_party:python_requirements.txt",
)

load("@tflm_pip_deps//:requirements.bzl", "install_deps")
install_deps()

model_repos()
