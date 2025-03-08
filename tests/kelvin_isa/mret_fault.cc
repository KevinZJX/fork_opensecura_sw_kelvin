// Copyright 2025 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <cstdint>

extern "C" {

void isr_wrapper(void);
__attribute__((naked)) void isr_wrapper() {
  asm volatile(
      "csrr t0, mepc \n"
      "addi t0, t0, 2 \n"
      "csrw mepc, t0 \n"
      "mret \n");
}
void bad_isr(void);
__attribute__((naked)) void bad_isr() { asm volatile("ebreak \n"); }

__attribute__((naked, aligned(256))) void isr_vector_table() {
  asm volatile(
      "j isr_wrapper \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n"
      "j bad_isr \n");
}

int main(int argc, char** argv) {
  asm volatile("csrw mtvec, %0" ::"rK"((uint32_t)(&isr_vector_table)));
  asm volatile("ecall");
  return 0;
}

}  // extern "C"