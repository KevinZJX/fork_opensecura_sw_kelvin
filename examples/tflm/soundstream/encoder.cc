// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "examples/tflm/soundstream/encoder.h"

#include "examples/tflm/soundstream/encoder_non_stream_q16x8_b64_io_int16_tflite.h"
#include "examples/tflm/soundstream/encoder_streaming_q16x8_b64_io_int16_tflite.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/recording_micro_allocator.h"

namespace kelvin::soundstream::encoder {

constexpr unsigned int kNonStreamingOpCount = 6;
constexpr unsigned int kStreamingOpCount = 13;
// Not sure how to get a good upper bound on this one, so arbitrarily chosen.
constexpr unsigned int kStreamingVariablesCount = 40;

template <bool kStreaming>
class EncoderImpl : public Encoder {
 public:
  static Encoder* Setup(const uint8_t* model_data, uint8_t* tensor_arena,
                        size_t tensor_arena_size) {
    auto* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      return nullptr;
    }

    EncoderImpl* e = new EncoderImpl(model, tensor_arena, tensor_arena_size);

    TfLiteStatus allocate_status = e->interpreter()->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      MicroPrintf("Failed to allocate decoder's tensors");
      return nullptr;
    }
    return e;
  }
  tflite::MicroInterpreter* interpreter() { return &interpreter_; }

 private:
  EncoderImpl(const tflite::Model* model, uint8_t* tensor_arena,
              size_t tensor_arena_size)
      : resolver_(CreateResolver()),
        allocator_(tflite::RecordingMicroAllocator::Create(tensor_arena,
                                                           tensor_arena_size)),
        variables_(tflite::MicroResourceVariables::Create(
            allocator_.get(), kStreamingVariablesCount)),
        interpreter_(model, resolver_, allocator_.get(), variables_.get()) {}

  static constexpr int kOpCount =
      kStreaming ? kStreamingOpCount : kStreamingOpCount;
  static inline tflite::MicroMutableOpResolver<kOpCount> CreateResolver() {
    tflite::MicroMutableOpResolver<kOpCount> resolver;
    resolver.AddReshape();
    resolver.AddPad();
    resolver.AddConv2D();
    resolver.AddLeakyRelu();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    if (kStreaming) {
      resolver.AddCallOnce();
      resolver.AddVarHandle();
      resolver.AddReadVariable();
      resolver.AddConcatenation();
      resolver.AddStridedSlice();
      resolver.AddAssignVariable();
      resolver.AddQuantize();
    }
    return resolver;
  }
  const tflite::MicroMutableOpResolver<kOpCount> resolver_;
  // Created in the arena
  std::unique_ptr<tflite::RecordingMicroAllocator> allocator_;
  // Created in the arena
  std::unique_ptr<tflite::MicroResourceVariables> variables_;
  tflite::MicroInterpreter interpreter_;
};

// Two separate methods to construct streaming vs non-streaming, so that the
// compiler can eliminate one if it's unused. Perhaps with LTO we could combine
// them together.
std::unique_ptr<Encoder> SetupStreaming(uint8_t* tensor_arena,
                                        size_t tensor_arena_size) {
  return std::unique_ptr<Encoder>(EncoderImpl<true>::Setup(
      g__encoder_streaming_q16x8_b64_io_int16_model_data, tensor_arena,
      tensor_arena_size));
}
std::unique_ptr<Encoder> Setup(uint8_t* tensor_arena,
                               size_t tensor_arena_size) {
  return std::unique_ptr<Encoder>(EncoderImpl<false>::Setup(
      g__encoder_non_stream_q16x8_b64_io_int16_model_data, tensor_arena,
      tensor_arena_size));
}

}  // namespace kelvin::soundstream::encoder
