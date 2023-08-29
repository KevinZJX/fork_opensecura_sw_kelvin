// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLES_TFLM_SOUNDSTREAM_DECODER_H_
#define EXAMPLES_TFLM_SOUNDSTREAM_DECODER_H_

#include <cstddef>
#include <memory>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"

namespace kelvin::soundstream::decoder {
// RecordingMicroAllocator on desktop recorded 94512 bytes of allocation.
constexpr size_t kTensorArenaSizeBytes = 96 * 1024;
// RecordingMicroAllocator on desktop recorded 143296 bytes of allocation.
constexpr size_t kTensorArenaStreamingSizeBytes = 168 * 1024;

class Decoder {
 public:
  virtual tflite::MicroInterpreter* interpreter() = 0;
};

std::unique_ptr<Decoder> Setup(uint8_t* tensor_arena, size_t tensor_arena_size);
std::unique_ptr<Decoder> SetupStreaming(uint8_t* tensor_arena,
                                        size_t tensor_arena_size);
}  // namespace kelvin::soundstream::decoder

#endif  // EXAMPLES_TFLM_SOUNDSTREAM_DECODER_H_
