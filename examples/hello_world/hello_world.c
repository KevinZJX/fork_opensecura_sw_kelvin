// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
// A Simple kelvin program.

#include <stddef.h>
#include <stdint.h>

// This header is included to ensure that it can
// be referenced from a C file.
#include "crt/kelvin.h"

typedef struct {
  uint32_t return_code;  // Populated in kelvin_start.S.
  uint32_t output_ptr;
  uint32_t length;
} OutputHeader;

__attribute__((section(".model_output_header"))) OutputHeader output_header = {
    .output_ptr = 0,
    .length = 0,
};

__attribute__((section(".model_output"))) uint32_t output;

int main(int argc, char *argv[]) {
  const uint32_t kDataSize = 0x1000 / sizeof(uint32_t);
  uint32_t data[kDataSize];

  for (int i = 0; i < sizeof(data) / sizeof(uint32_t); ++i) {
    data[i] = i;
  }

  for (int i = 0; i < sizeof(data) / sizeof(uint32_t); ++i) {
    data[i] += 1;
  }

  // Setup output.
  output_header.length = sizeof(output);
  output = data[kDataSize - 1];
  output_header.output_ptr = (uint32_t)&output;

  return 0;
}
