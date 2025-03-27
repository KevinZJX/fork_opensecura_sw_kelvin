# Copyright 2024 Google LLC
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at#
#     http://www.apache.org/licenses/LICENSE-2.0#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load("@bazel_skylib//rules:common_settings.bzl", "bool_flag")

package(default_visibility = ["//visibility:public"])

bool_flag(
    name = "link_tcm",
    build_setting_default = False,
)

bool_flag(
    name = "link_tcm_highmem",
    build_setting_default = False,
)

config_setting(
    name = "link_tcm_config",
    flag_values = {
        ":link_tcm": "True",
    },
)

config_setting(
    name = "link_tcm_highmem_config",
    flag_values = {
        ":link_tcm_highmem": "True",
    },
)
