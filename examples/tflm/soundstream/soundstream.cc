// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "examples/tflm/soundstream/best_of_times_s16_decoded.h"
#include "examples/tflm/soundstream/best_of_times_s16_encoded.h"
#include "examples/tflm/soundstream/best_of_times_s16_wav.h"
#include "examples/tflm/soundstream/decoder_non_stream_q16x8_b64_io_int16_tflite.h"
#include "examples/tflm/soundstream/encoder_non_stream_q16x8_b64_io_int16_tflite.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace {
const tflite::Model *encoder_model = nullptr;
const tflite::Model *decoder_model = nullptr;
tflite::MicroInterpreter *encoder_interpreter = nullptr;
tflite::MicroInterpreter *decoder_interpreter = nullptr;
constexpr int kTensorArenaSize =
    96 * 1024;
uint8_t encoder_tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
uint8_t decoder_tensor_arena[kTensorArenaSize] __attribute__((aligned(16)));
}  // namespace

int main(int argc, char **argv) {
  encoder_model =
      tflite::GetModel(g__encoder_non_stream_q16x8_b64_io_int16_model_data);
  if (encoder_model->version() != TFLITE_SCHEMA_VERSION) {
    return 1;
  }
  decoder_model =
      tflite::GetModel(g__decoder_non_stream_q16x8_b64_io_int16_model_data);
  if (decoder_model->version() != TFLITE_SCHEMA_VERSION) {
    return 1;
  }

  static tflite::MicroMutableOpResolver<6> encoder_resolver{};
  encoder_resolver.AddReshape();
  encoder_resolver.AddPad();
  encoder_resolver.AddConv2D();
  encoder_resolver.AddLeakyRelu();
  encoder_resolver.AddDepthwiseConv2D();
  encoder_resolver.AddAdd();

  static tflite::MicroMutableOpResolver<11> decoder_resolver{};
  decoder_resolver.AddReshape();
  decoder_resolver.AddPad();
  decoder_resolver.AddConv2D();
  decoder_resolver.AddLeakyRelu();
  decoder_resolver.AddSplit();
  decoder_resolver.AddTransposeConv();
  decoder_resolver.AddStridedSlice();
  decoder_resolver.AddConcatenation();
  decoder_resolver.AddDepthwiseConv2D();
  decoder_resolver.AddAdd();
  decoder_resolver.AddQuantize();

  static tflite::MicroInterpreter encoder_static_interpreter(
      encoder_model, encoder_resolver, encoder_tensor_arena, kTensorArenaSize);
  encoder_interpreter = &encoder_static_interpreter;

  static tflite::MicroInterpreter decoder_static_interpreter(
      decoder_model, decoder_resolver, decoder_tensor_arena, kTensorArenaSize);
  decoder_interpreter = &decoder_static_interpreter;

  TfLiteStatus allocate_status = encoder_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("Failed to allocate encoder's tensors");
    return -1;
  }
  allocate_status = decoder_interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("Failed to allocate decoder's tensors");
    return -1;
  }

  TfLiteTensor *encoder_input = encoder_interpreter->input(0);
  TfLiteTensor *encoder_output = encoder_interpreter->output(0);
  TfLiteTensor *decoder_input = decoder_interpreter->input(0);
  TfLiteTensor *decoder_output = decoder_interpreter->output(0);

  int invocation_count =
      (g_best_of_times_s16_audio_data_size * sizeof(int16_t)) /
      encoder_input->bytes;
  for (int i = 0; i < invocation_count; ++i) {
    MicroPrintf("Invocation %d of %d", i, invocation_count);
    memcpy(encoder_input->data.uint8,
           g_best_of_times_s16_audio_data +
               ((i * encoder_input->bytes) / sizeof(int16_t)),
           encoder_input->bytes);
    TfLiteStatus invoke_status = encoder_interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Failed to invoke encoder");
      return -1;
    }
    if (memcmp(encoder_output->data.uint8,
               g_best_of_times_s16_encoded + (i * encoder_output->bytes),
               encoder_output->bytes)) {
      MicroPrintf("Encoder output mismatches reference");
      return -1;
    }

    memcpy(decoder_input->data.uint8, encoder_output->data.uint8,
           decoder_input->bytes);
    invoke_status = decoder_interpreter->Invoke();
    if (invoke_status != kTfLiteOk) {
      MicroPrintf("Failed to invoke decoder");
      return -1;
    }
    if (memcmp(decoder_output->data.uint8,
               g_best_of_times_s16_decoded + (i * decoder_output->bytes),
               decoder_output->bytes)) {
      MicroPrintf("Decoder output mismatches reference");
      return -1;
    }
  }

  return 0;
}
