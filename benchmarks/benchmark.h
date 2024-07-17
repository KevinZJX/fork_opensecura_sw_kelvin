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

#ifndef BENCHMARKS_BENCHMARK_H_
#define BENCHMARKS_BENCHMARK_H_

#define ML_RUN_INDICATOR_IO 16
#define ML_TOGGLE_PER_INF_IO 17

typedef struct {
  uint32_t return_code;
  uint32_t iterations;
  uint64_t cycles;
  uint32_t mismatch_count;
  uint32_t gpio_toggle_per_inference;
} BenchmarkOutputHeader;

#endif  // BENCHMARKS_BENCHMARK_H_
