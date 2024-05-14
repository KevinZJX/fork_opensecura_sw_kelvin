# Copyright 2023 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

load(
    "@crt//config:features.bzl",
    "CPP_ALL_COMPILE_ACTIONS",
    "C_ALL_COMPILE_ACTIONS",
    "LD_ALL_ACTIONS",
    "feature",
    "feature_set",
    "flag_group",
    "flag_set",
)

package(default_visibility = ["//visibility:public"])

feature(
    name = "architecture",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = CPP_ALL_COMPILE_ACTIONS + C_ALL_COMPILE_ACTIONS + LD_ALL_ACTIONS,
            flag_groups = [
                flag_group(
                    flags = [
                        "-march=ARCHITECTURE",
                        "-mabi=ABI",
                        "-mcmodel=CMODEL",
                    ],
                ),
            ],
        ),
    ],
)

feature(
    name = "sys_spec",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = CPP_ALL_COMPILE_ACTIONS,
            flag_groups = [
                flag_group(
                    flags = [
                        "-march=ARCHITECTURE",
                        "-mabi=ABI",
                        "-mcmodel=CMODEL",
                        "-nostdlib",
                    ],
                ),
            ],
        ),
        flag_set(
            actions = LD_ALL_ACTIONS,
            flag_groups = [
                flag_group(
                    flags = [
                        "--specs=nano.specs",
                        "-lm_nano",
                        "-lc_nano",
                        "-lg_nano",
                        "-lgcc",
                        "-nostartfiles",
                    ],
                ),
            ],
        ),
    ],
)

feature(
    name = "all_warnings",
    enabled = True,
    flag_sets = [
        flag_set(
            actions = CPP_ALL_COMPILE_ACTIONS + C_ALL_COMPILE_ACTIONS,
            flag_groups = [
                flag_group(
                    flags = [
                        "-Wall",
                    ],
                ),
            ],
        ),
    ],
)

feature(
    name = "all_warnings_as_errors",
    enabled = False,
    flag_sets = [
        flag_set(
            actions = CPP_ALL_COMPILE_ACTIONS + C_ALL_COMPILE_ACTIONS,
            flag_groups = [
                flag_group(
                    flags = ["-Werror"],
                ),
            ],
        ),
    ],
)

feature(
    name = "fastbuild",
    enabled = False,
    flag_sets = [
        flag_set(
            actions = CPP_ALL_COMPILE_ACTIONS + C_ALL_COMPILE_ACTIONS,
            flag_groups = [
                flag_group(
                    flags = [
                        "-O2",
                        "-g",
                    ],
                ),
            ],
        ),
    ],
    provides = ["compilation_mode"],
)

feature_set(
    name = "rv32im",
    feature = [
        ":all_warnings",
        ":all_warnings_as_errors",
        ":architecture",
        ":fastbuild",
        ":sys_spec",
        "@crt//features/common:includes",
        "@crt//features/common:reproducible",
        "@crt//features/common:symbol_garbage_collection",
        "@crt//features/embedded:cc_constructor_destructor",
        "@crt//features/embedded:exceptions",
        "@crt//features/embedded:runtime_type_information",
    ],
)
