// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "examples/tflm/soundstream/decoder.h"

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
    kelvin::soundstream::decoder::kTensorArenaStreamingSizeBytes;
#else
constexpr size_t tensor_arena_size =
    kelvin::soundstream::decoder::kTensorArenaSizeBytes;
#endif
uint8_t decoder_tensor_arena[tensor_arena_size] __attribute__((aligned(64)));
}  // namespace

int main(int argc, char **argv) {
#if defined(STREAMING)
  auto decoder = kelvin::soundstream::decoder::SetupStreaming(
      decoder_tensor_arena, tensor_arena_size);
#else
  auto decoder = kelvin::soundstream::decoder::Setup(decoder_tensor_arena,
                                                     tensor_arena_size);
#endif
  if (!decoder) {
    MicroPrintf("Unable to construct decoder");
    return -1;
  }

  TfLiteTensor *decoder_input = decoder->interpreter()->input(0);
  TfLiteTensor *decoder_output = decoder->interpreter()->output(0);

  memset(decoder_input->data.uint8, 0, decoder_input->bytes);
  TfLiteStatus invoke_status = decoder->interpreter()->Invoke();
  if (invoke_status != kTfLiteOk) {
    MicroPrintf("Failed to invoke decoder");
    return -1;
  }

  output_header.length = decoder_output->bytes;
  output_header.output_ptr =
      reinterpret_cast<uint32_t>(decoder_output->data.uint8);
  return 0;
}
