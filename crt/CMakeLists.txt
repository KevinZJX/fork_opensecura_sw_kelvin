# Copyright 2023 Google LLC
# Licensed under the Apache License, Version 2.0, see LICENSE for details.
# SPDX-License-Identifier: Apache-2.0
#

enable_language(ASM)


include_directories(BEFORE SYSTEM $ENV{ROOTDIR}/sw/kelvin)

# Build Kelvin
add_library(kelvin INTERFACE)
add_library(kelvin_intrinsic STATIC)
target_sources(kelvin_intrinsic
    PRIVATE
      crt.S
      kelvin_start.S
      kelvin_gloss.cc
)

target_include_directories(kelvin_intrinsic PUBLIC "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_libraries(kelvin
INTERFACE
  kelvin_intrinsic
)

target_include_directories(kelvin INTERFACE "${CMAKE_CURRENT_SOURCE_DIR}")

target_link_options(kelvin
    INTERFACE
      -Wl,--whole-archive ${CMAKE_CURRENT_BINARY_DIR}/libkelvin_intrinsic.a -Wl,--no-whole-archive,--no-warn-rwx-segments
)
