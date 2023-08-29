// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "examples/tflm/soundstream/best_of_times_s16_wav.h"
#include "examples/tflm/soundstream/decoder.h"
#include "examples/tflm/soundstream/encoder.h"

#if defined(STREAMING)
#include "examples/tflm/soundstream/best_of_times_s16_decoded_streaming.h"
#include "examples/tflm/soundstream/best_of_times_s16_encoded_streaming.h"
const unsigned char *reference_decoded = g_best_of_times_s16_decoded_streaming;
const unsigned char *reference_encoded = g_best_of_times_s16_encoded_streaming;
#else
#include "examples/tflm/soundstream/best_of_times_s16_decoded.h"
#include "examples/tflm/soundstream/best_of_times_s16_encoded.h"
const unsigned char *reference_decoded = g_best_of_times_s16_decoded;
const unsigned char *reference_encoded = g_best_of_times_s16_encoded;
#endif

namespace {
#if defined(STREAMING)
constexpr size_t decoder_tensor_arena_size =
    kelvin::soundstream::decoder::kTensorArenaStreamingSizeBytes;
constexpr size_t encoder_tensor_arena_size =
    kelvin::soundstream::encoder::kTensorArenaStreamingSizeBytes;
#else
constexpr size_t decoder_tensor_arena_size =
    kelvin::soundstream::decoder::kTensorArenaSizeBytes;
constexpr size_t encoder_tensor_arena_size =
    kelvin::soundstream::encoder::kTensorArenaSizeBytes;
#endif
uint8_t encoder_tensor_arena[encoder_tensor_arena_size]
    __attribute__((aligned(64)));
uint8_t decoder_tensor_arena[decoder_tensor_arena_size]
    __attribute__((aligned(64)));
}  // namespace

int main(int argc, char **argv) {
#if defined(STREAMING)
  auto encoder = kelvin::soundstream::encoder::SetupStreaming(
      encoder_tensor_arena, encoder_tensor_arena_size);
#else
  auto encoder = kelvin::soundstream::encoder::Setup(encoder_tensor_arena,
                                                     encoder_tensor_arena_size);
#endif
  if (!encoder) {
    MicroPrintf("Unable to construct encoder");
    return -1;
  }

#if defined(STREAMING)
  auto decoder = kelvin::soundstream::decoder::SetupStreaming(
      decoder_tensor_arena, decoder_tensor_arena_size);
#else
  auto decoder = kelvin::soundstream::decoder::Setup(decoder_tensor_arena,
                                                     decoder_tensor_arena_size);
#endif
  if (!decoder) {
    MicroPrintf("Unable to construct decoder");
    return -1;
  }

  TfLiteTensor *encoder_input = encoder->interpreter()->input(0);
  TfLiteTensor *encoder_output = encoder->interpreter()->output(0);
  TfLiteTensor *decoder_input = decoder->interpreter()->input(0);
  TfLiteTensor *decoder_output = decoder->interpreter()->output(0);

  int invocation_count =
      (g_best_of_times_s16_audio_data_size * sizeof(int16_t)) /
      encoder_input->bytes;
  for (int i = 0; i < invocation_count; ++i) {
    MicroPrintf("Invocation %d of %d", i, invocation_count);
    memcpy(encoder_input->data.uint8,
           g_best_of_times_s16_audio_data +
               ((i * encoder_input->bytes) / sizeof(int16_t)),
           encoder_input->bytes);
    TfLiteStatus invoke_status = encoder->interpreter()->Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Failed to invoke encoder");
      return -1;
    }
    if (memcmp(encoder_output->data.uint8,
               reference_encoded + (i * encoder_output->bytes),
               encoder_output->bytes)) {
      MicroPrintf("Encoder output mismatches reference");
      return -1;
    }

    memcpy(decoder_input->data.uint8, encoder_output->data.uint8,
           decoder_input->bytes);
    invoke_status = decoder->interpreter()->Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Failed to invoke decoder");
      return -1;
    }
    if (memcmp(decoder_output->data.uint8,
               reference_decoded + (i * decoder_output->bytes),
               decoder_output->bytes)) {
      MicroPrintf("Decoder output mismatches reference");
      return -1;
    }
  }

  return 0;
}
