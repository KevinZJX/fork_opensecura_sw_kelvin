// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#include "examples/tflm/soundstream/decoder.h"

#include "examples/tflm/soundstream/decoder_non_stream_q16x8_b64_io_int16_tflite.h"
#include "examples/tflm/soundstream/decoder_streaming_q16x8_b64_io_int16_tflite.h"

namespace kelvin::soundstream::decoder {

constexpr unsigned int kNonStreamingOpCount = 11;
constexpr unsigned int kStreamingOpCount = 16;
// Not sure how to get a good upper bound on this one, so arbitrarily chosen.
constexpr unsigned int kStreamingVariablesCount = 40;

template <bool kStreaming>
class DecoderImpl : public Decoder {
 public:
  static Decoder* Setup(const uint8_t* model_data, uint8_t* tensor_arena,
                        size_t tensor_arena_size) {
    auto* model = tflite::GetModel(model_data);
    if (model->version() != TFLITE_SCHEMA_VERSION) {
      return nullptr;
    }

    DecoderImpl* d = new DecoderImpl(model, tensor_arena, tensor_arena_size);

    TfLiteStatus allocate_status = d->interpreter()->AllocateTensors();
    if (allocate_status != kTfLiteOk) {
      MicroPrintf("Failed to allocate decoder's tensors");
      return nullptr;
    }
    return d;
  }
  tflite::MicroInterpreter* interpreter() { return &interpreter_; }

 private:
  DecoderImpl(const tflite::Model* model, uint8_t* tensor_arena,
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
    resolver.AddSplit();
    resolver.AddTransposeConv();
    resolver.AddStridedSlice();
    resolver.AddConcatenation();
    resolver.AddDepthwiseConv2D();
    resolver.AddAdd();
    resolver.AddQuantize();
    if (kStreaming) {
      resolver.AddCallOnce();
      resolver.AddVarHandle();
      resolver.AddReadVariable();
      resolver.AddAssignVariable();
      resolver.AddSub();
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
std::unique_ptr<Decoder> SetupStreaming(uint8_t* tensor_arena,
                                        size_t tensor_arena_size) {
  return std::unique_ptr<Decoder>(DecoderImpl<true>::Setup(
      g__decoder_streaming_q16x8_b64_io_int16_model_data, tensor_arena,
      tensor_arena_size));
}
std::unique_ptr<Decoder> Setup(uint8_t* tensor_arena,
                               size_t tensor_arena_size) {
  return std::unique_ptr<Decoder>(DecoderImpl<false>::Setup(
      g__decoder_non_stream_q16x8_b64_io_int16_model_data, tensor_arena,
      tensor_arena_size));
}

}  // namespace kelvin::soundstream::decoder
