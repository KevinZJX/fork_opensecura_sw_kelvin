# Kelvin SW Repository

This project contains the BSP to build the SW artifact that can run on the
Kelvin core, and integrated as part of the Shodan repository.

The project supports two build systems -- Bazel and CMake -- for OSS integration
reasons. Bazel is used by [TFLM](https://github.com/tensorflow/tflite-micro)
flow, while CMake is the build system for [IREE](https://github.com/openxla/iree).

## Prerequisite

If you get this project from Project Shodan manifest, you are all set. If not,
you need to have following projects as well to build the project successfully.

* Kelvin crosscompile toolchain: Under `<dir>/cache/toolchain_kelvin`

This project needs to be at `<dir>/sw/kelvin`.

## Code structure

* build_tools: Build tool/rules for both Bazel and CMake
* crt: Kelvin BSP
* examples: Source code to build Kelvin SW artifacts.
* platforms: Crosscompile platform setup for Bazel.
* third_party: Third party repositories for Bazel.
* toolchains: Crosscomple toolchain setup for Bazel.
* host_tools: host tool to generate the intrinsic header and toolchain op files

## Build the project

### Bazel

The project uses Bazel 5.1.1, to align with
[OpenTitan](https://github.com/lowRISC/opentitan) build system requirements.

```bash
bazel build //...
```

To run the unit tests (with the kelvin_sim ISS)

```bash
bazel test --test_env=ROOTDIR=${ROOTDIR} //...
```

### CMake

```note
TODO: Add CMake flow
```

## Run the executable

The binaries can be simulated with the kelvin simulator, located at
`<dir>/sim/kelvin`.

```note
sim_kelvin <elf location>
```

Load the generated `.bin` binaries to the FPGA emulator/Renode simulator.

