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

load("@crt//config:device.bzl", "device_config")

DEVICES = [
    device_config(
        name = "kelvin",
        architecture = "rv32im",
        feature_set = "//platforms/riscv32/features:rv32im",
        constraints = [
            "//platforms/cpu:kelvin",
            "@platforms//os:none",
        ],
        substitutions = {
            "ARCHITECTURE": "rv32i2p1m_zicsr_zifencei_zbb",
            "ABI": "ilp32",
            "CMODEL": "medany",
            "[STACK_PROTECTOR]": "-fstack-protector-strong",
        },
    ),
]
