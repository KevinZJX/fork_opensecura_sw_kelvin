#!/bin/bash

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

