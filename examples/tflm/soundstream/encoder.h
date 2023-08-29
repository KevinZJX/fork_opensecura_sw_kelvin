// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef EXAMPLES_TFLM_SOUNDSTREAM_ENCODER_H_
#define EXAMPLES_TFLM_SOUNDSTREAM_ENCODER_H_

#include <cstddef>
#include <memory>

#include "tensorflow/lite/micro/micro_interpreter.h"

namespace kelvin::soundstream::encoder {
// RecordingMicroAllocator on desktop recorded 90064 bytes of allocation.
constexpr size_t kTensorArenaSizeBytes = 96 * 1024;
// RecordingMicroAllocator on desktop recorded 147328 bytes of allocation.
constexpr size_t kTensorArenaStreamingSizeBytes = 168 * 1024;

struct Encoder {
  virtual tflite::MicroInterpreter* interpreter() = 0;
};

std::unique_ptr<Encoder> Setup(uint8_t* tensor_arena, size_t tensor_arena_size);
std::unique_ptr<Encoder> SetupStreaming(uint8_t* tensor_arena,
                                        size_t tensor_arena_size);
}  // namespace kelvin::soundstream::encoder

#endif  // EXAMPLES_TFLM_SOUNDSTREAM_ENCODER_H_
