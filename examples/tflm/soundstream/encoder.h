#ifndef EXAMPLES_TFLM_SOUNDSTREAM_ENCODER_H_
#define EXAMPLES_TFLM_SOUNDSTREAM_ENCODER_H_

#include <cstddef>
#include <memory>
#include <optional>

#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"

namespace kelvin::soundstream::encoder {
constexpr size_t kTensorArenaSizeBytes = 96 * 1024;
struct Encoder {
  std::unique_ptr<tflite::MicroInterpreter> interpreter;
  std::unique_ptr<tflite::MicroMutableOpResolver<6>> resolver;
};
std::optional<Encoder> Setup(uint8_t* tensor_arena);
}  // namespace kelvin::soundstream::encoder

#endif  // EXAMPLES_TFLM_SOUNDSTREAM_ENCODER_H_
