#ifndef EXAMPLES_TFLM_SOUNDSTREAM_DECODER_H_
#define EXAMPLES_TFLM_SOUNDSTREAM_DECODER_H_

#include <cstddef>
#include <memory>
#include <optional>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace kelvin::soundstream::decoder {
constexpr size_t kTensorArenaSizeBytes = 96 * 1024;
struct Decoder {
  std::unique_ptr<tflite::MicroInterpreter> interpreter;
  std::unique_ptr<tflite::MicroMutableOpResolver<11>> resolver;
};
std::optional<Decoder> Setup(uint8_t* tensor_arena);
}  // namespace kelvin::soundstream::decoder

#endif  // EXAMPLES_TFLM_SOUNDSTREAM_DECODER_H_
