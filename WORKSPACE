# Copyright 2023 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at#
#     http://www.apache.org/licenses/LICENSE-2.0#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#

workspace(name = "kelvin_sw")

load("//build_tools/bazel:repos.bzl", "kelvin_sw_repos", "tflm_repos")

kelvin_sw_repos()

# Register Kelvin toolchain

load("//platforms:registration.bzl", "kelvin_register_toolchain")

kelvin_register_toolchain()

tflm_repos()
load("@rules_python//python:repositories.bzl", "py_repositories")
py_repositories()

load("@rules_python//python:repositories.bzl", "python_register_toolchains")
python_register_toolchains(
     name = "python3",
     python_version = "3.9",
)

load("@tflite-micro//tensorflow:workspace.bzl", "tf_repositories")

tf_repositories()

load("@rules_python//python:pip.bzl", "pip_parse")
pip_parse(
    name = "tflm_pip_deps",
    python_interpreter_target = "@python3_x86_64-unknown-linux-gnu//:python",
    requirements_lock = "@tflite-micro//third_party:python_requirements.txt",
)

load("@tflm_pip_deps//:requirements.bzl", "install_deps")
install_deps()
