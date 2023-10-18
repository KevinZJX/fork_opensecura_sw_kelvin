#!/bin/bash
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


function print_usage {
  echo "Usage: test_runner.sh <elf location>"
}

if [[ $1 == "--help" ]]; then
  print_usage
fi

if [[ -z ${ROOTDIR} ]]; then
  echo "Please run \"source build/setup.sh\" first"
  exit 1
fi

if [[ ! -f ${ROOTDIR}/out/kelvin/sim/kelvin_sim ]]; then
  echo "Please run \"m kelvin_sim\" first"
  exit 1
fi

if (( $# != 1 )); then
  print_usage
  exit 1
fi

ELF_FILE=$(realpath $1)
SIM_OUT=$(${ROOTDIR}/out/kelvin/sim/kelvin_sim "${ELF_FILE}")
echo "${SIM_OUT}"
if [[ ! "${SIM_OUT}" == *"Program exits properly"* ]]; then
  exit 1
fi

