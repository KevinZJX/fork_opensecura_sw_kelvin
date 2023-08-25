#include "examples/tflm/soundstream/encoder.h"

#include "examples/tflm/soundstream/encoder_non_stream_q16x8_b64_io_int16_tflite.h"

namespace kelvin::soundstream::encoder {
std::optional<Encoder> Setup(uint8_t* tensor_arena) {
  auto* model =
      tflite::GetModel(g__encoder_non_stream_q16x8_b64_io_int16_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return {};
  }

  Encoder e;
  e.resolver = std::make_unique<tflite::MicroMutableOpResolver<6>>();
  e.resolver->AddReshape();
  e.resolver->AddPad();
  e.resolver->AddConv2D();
  e.resolver->AddLeakyRelu();
  e.resolver->AddDepthwiseConv2D();
  e.resolver->AddAdd();

  e.interpreter = std::make_unique<tflite::MicroInterpreter>(
      model, *e.resolver, tensor_arena, kTensorArenaSizeBytes);

  TfLiteStatus allocate_status = e.interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("Failed to allocate encoder's tensors");
    return {};
  }
  return e;
}
}  // namespace kelvin::soundstream::encoder
