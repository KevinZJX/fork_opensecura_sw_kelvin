/*
 * Copyright 2024 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *      http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <memory>

#include "crt/kelvin.h"
#include "crt/log.h"
#include "benchmarks/benchmark.h"
#include "tensorflow/lite/micro/micro_interpreter.h"
#include "tensorflow/lite/micro/micro_log.h"
#include "tensorflow/lite/micro/micro_mutable_op_resolver.h"
#include "tensorflow/lite/schema/schema_generated.h"

#define STRINGIZE(x) #x
#define STR(x) STRINGIZE(x)

// In order to include the model data generate from Bazel, include the header
// using the name passed as a macro.
#define MODEL_HEADER_DIRECTORY benchmarks/
#define MODEL_HEADER_TYPE _model.h
#define MODEL_HEADER STR(MODEL_HEADER_DIRECTORY BENCHMARK_NAME MODEL_HEADER_TYPE)
#include MODEL_HEADER

namespace {
constexpr int kTensorArenaSize = 1024 * 1024;
uint8_t g_tensor_arena[kTensorArenaSize] __attribute__((aligned(64)));

__attribute__((section(".model_output_header"))) BenchmarkOutputHeader output_header = {
    .return_code = 0, // Set by kelvin_start based on return value in main.
    .iterations = 0,
    .cycles = 0,
};

inline uint64_t mcycle_read(void) {
  uint32_t cycle_low = 0;
  uint32_t cycle_high = 0;
  uint32_t cycle_high_2 = 0;
  asm volatile(
      "1:"
      "  csrr %0, mcycleh;"  // Read `mcycleh`.
      "  csrr %1, mcycle;"   // Read `mcycle`.
      "  csrr %2, mcycleh;"  // Read `mcycleh` again.
      "  bne  %0, %2, 1b;"
      : "=r"(cycle_high), "=r"(cycle_low), "=r"(cycle_high_2)
      :);
  return static_cast<uint64_t>(cycle_high) << 32 | cycle_low;
}

// This includes all ops currently used in the Kelvin model suite. More can be added.
constexpr int kAllOpsNum = 22;
std::unique_ptr<tflite::MicroMutableOpResolver<kAllOpsNum>> GetAllOpsResolver() {
  tflite::MicroMutableOpResolver<kAllOpsNum> resolver;
  resolver.AddAveragePool2D();
  resolver.AddMaxPool2D();
  resolver.AddConv2D();
  resolver.AddConcatenation();
  resolver.AddDepthwiseConv2D();
  resolver.AddDequantize();
  resolver.AddQuantize();
  resolver.AddReshape();
  resolver.AddSoftmax();
  resolver.AddCallOnce();
  resolver.AddVarHandle();
  resolver.AddReadVariable();
  resolver.AddAssignVariable();
  resolver.AddLogistic();
  resolver.AddStridedSlice();
  resolver.AddFullyConnected();
  resolver.AddPad();
  resolver.AddLeakyRelu();
  resolver.AddSplit();
  resolver.AddTransposeConv();
  resolver.AddAdd();
  resolver.AddSub();
  return std::make_unique<tflite::MicroMutableOpResolver<kAllOpsNum>>(resolver);
}

void _print64(const char* header, uint64_t number) {
  uint32_t number_low = number & 0xFFFFFFFF;
  uint32_t number_hi = number >> 32;
  LOG_INFO("%s: 0x%08lx%08lx", header, number_hi, number_low);
}

constexpr int kSuccess = 0;
constexpr int kAllocatonFailed = -1;
constexpr int kInvokeFailed = -2;
} // namespace


int main(int argc, char **argv) {
  std::unique_ptr<tflite::MicroMutableOpResolver<kAllOpsNum>> resolver = GetAllOpsResolver();

  const auto* model = tflite::GetModel(g_benchmark_model_data);

  uint8_t variable_arena[2048];
  tflite::MicroAllocator *variable_allocator =
      tflite::MicroAllocator::Create(variable_arena, 1024);
  tflite::MicroResourceVariables *resource_variables =
      tflite::MicroResourceVariables::Create(variable_allocator, 20);
  std::unique_ptr<tflite::MicroInterpreter> interpreter = std::make_unique<tflite::MicroInterpreter>(
      model, *resolver.get(), g_tensor_arena, kTensorArenaSize, resource_variables);

  // Run inference outside of benchmark to intialize model.
  if (interpreter->AllocateTensors() != kTfLiteOk) {
    return kAllocatonFailed;
  }
  TfLiteTensor* input = interpreter->input(0);

  // Set input tensor to zero for first inference, subsequent runs
  // will run on output tensor data (since the memory is shared).
  memset(tflite::GetTensorData<uint8_t>(input), 0, input->bytes);
  if (interpreter->Invoke() != kTfLiteOk) {
    return kInvokeFailed;
  }

  LOG_INFO("========== Begin Benchmark (%s) ==========", STR(BENCHMARK_NAME));
  uint64_t begin = mcycle_read();

  // TODO(michaelbrooks): Possibly set/verify test data?
  for (int i = 0; i < ITERATIONS; ++i) {
    interpreter->Invoke();
  }
  uint64_t end = mcycle_read();
  uint64_t num_cycles = end - begin;
  // Stores benchmark information in output header for other cores to access.
  output_header.iterations = ITERATIONS;
  output_header.cycles = num_cycles;

  // If running on a simulator, print cycle information.
  uint64_t average_cycles = num_cycles / ITERATIONS;
  LOG_INFO("Iterations: %ld", output_header.iterations);
  _print64("Total Cycles: ", output_header.cycles);
  _print64("Average Cycles per Iteration: ", average_cycles);
  LOG_INFO("========== End Benchmark ==========");
  return kSuccess;
}
