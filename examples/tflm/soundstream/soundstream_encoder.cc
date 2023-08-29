// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "examples/tflm/soundstream/encoder.h"

typedef struct {
  uint32_t return_code;  // Populated in kelvin_start
  uint32_t output_ptr;
  uint32_t length;
} OutputHeader;

__attribute__((section(".model_output_header"))) OutputHeader output_header = {
    .output_ptr = 0,
    .length = 0,
};

namespace {
#if defined(STREAMING)
constexpr size_t tensor_arena_size =
    kelvin::soundstream::encoder::kTensorArenaStreamingSizeBytes;
#else
constexpr size_t tensor_arena_size =
    kelvin::soundstream::encoder::kTensorArenaSizeBytes;
#endif
uint8_t tensor_arena[tensor_arena_size] __attribute__((aligned(64)));
}  // namespace

int main(int argc, char **argv) {
#if defined(STREAMING)
  auto encoder = kelvin::soundstream::encoder::SetupStreaming(
      tensor_arena, tensor_arena_size);
#else
  auto encoder =
      kelvin::soundstream::encoder::Setup(tensor_arena, tensor_arena_size);
#endif
  if (!encoder) {
    MicroPrintf("Unable to construct encoder");
    return -1;
  }

  TfLiteTensor *encoder_input = encoder->interpreter()->input(0);
  TfLiteTensor *encoder_output = encoder->interpreter()->output(0);

  memset(encoder_input->data.uint8, 0, encoder_input->bytes);
  TfLiteStatus invoke_status = encoder->interpreter()->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Failed to invoke encoder");
    return -1;
  }

  output_header.length = encoder_output->bytes;
  output_header.output_ptr =
      reinterpret_cast<uint32_t>(encoder_output->data.uint8);
  return 0;
}
