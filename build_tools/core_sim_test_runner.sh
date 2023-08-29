#!/bin/bash
# Test runner for SystemC simulator. Note the input should be the .bin file.

function print_usage {
  echo "Usage: core_sim_test_runner.sh <bin location> [extra flags]"
}

if [[ $1 == "--help" ]]; then
  print_usage
fi

if [[ -z ${ROOTDIR} ]]; then
  echo "Please run \"source build/setup.sh\" first"
  exit 1
fi

CORE_SIM="${ROOTDIR}/out/kelvin/hw/bazel_out/core_sim"

if [[ ! -f ${CORE_SIM} ]]; then
  echo "Please run \"m kelvin_hw_sim\" first"
  exit 1
fi

if (( $# != 1 )); then
  print_usage
  exit 1
fi

BIN_FILE=$(realpath $1)
shift 1
SIM_OUT=$(${CORE_SIM} "${BIN_FILE}" $@)
RESULT=$?
echo "${SIM_OUT}"

exit ${RESULT}
