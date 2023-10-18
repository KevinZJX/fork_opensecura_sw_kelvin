/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "examples/tflm/person_detection/person_bmp.h"
#include "examples/tflm/person_detection/person_detect_tflite.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/micro/system_setup.h"

namespace {
const tflite::Model* model = nullptr;
tflite::MicroInterpreter* interpreter = nullptr;
constexpr int kTensorArenaSize = 96 * 1024;
uint8_t tensor_arena[kTensorArenaSize] __attribute__((aligned(64)));
}  // namespace

extern "C" int main(int argc, char** argv) {
  tflite::InitializeTarget();

  model = tflite::GetModel(g_person_detect_model_data);

  if (model->version() != TFLITE_SCHEMA_VERSION) {
    return 1;
  }

  static tflite::MicroMutableOpResolver<5> micro_op_resolver;
  micro_op_resolver.AddAveragePool2D();
  micro_op_resolver.AddConv2D();
  micro_op_resolver.AddDepthwiseConv2D();
  micro_op_resolver.AddReshape();
  micro_op_resolver.AddSoftmax();

  static tflite::MicroInterpreter static_interpreter(
      model, micro_op_resolver, tensor_arena, kTensorArenaSize);
  interpreter = &static_interpreter;

  TfLiteStatus allocate_status = interpreter->AllocateTensors();
  if (allocate_status != kTfLiteOk) {
    return 2;
  }

  TfLiteTensor* input = interpreter->input(0);
  TfLiteTensor* output = interpreter->output(0);

  memcpy(input->data.uint8, g_person_image_data, input->bytes);
  TfLiteStatus invoke_status = interpreter->Invoke();
  if (invoke_status != kTfLiteOk) {
    return 3;
  }

  int8_t person = output->data.int8[1];
  int8_t not_person = output->data.int8[0];
  MicroPrintf("person: %d not_person: %d", person, not_person);

  return 0;
}
