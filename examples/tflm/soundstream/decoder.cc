#include "examples/tflm/soundstream/decoder.h"

#include "examples/tflm/soundstream/decoder_non_stream_q16x8_b64_io_int16_tflite.h"

namespace kelvin::soundstream::decoder {
std::optional<Decoder> Setup(uint8_t* tensor_arena) {
  auto* model =
      tflite::GetModel(g__decoder_non_stream_q16x8_b64_io_int16_model_data);
  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return {};
  }

  Decoder d;
  d.resolver = std::make_unique<tflite::MicroMutableOpResolver<11>>();
  d.resolver->AddReshape();
  d.resolver->AddPad();
  d.resolver->AddConv2D();
  d.resolver->AddLeakyRelu();
  d.resolver->AddSplit();
  d.resolver->AddTransposeConv();
  d.resolver->AddStridedSlice();
  d.resolver->AddConcatenation();
  d.resolver->AddDepthwiseConv2D();
  d.resolver->AddAdd();
  d.resolver->AddQuantize();

  d.interpreter = std::make_unique<tflite::MicroInterpreter>(
      model, *d.resolver, tensor_arena, kTensorArenaSizeBytes);

  TfLiteStatus allocate_status = d.interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    MicroPrintf("Failed to allocate decoder's tensors");
    return {};
  }
  return d;
}
}  // namespace kelvin::soundstream::decoder
