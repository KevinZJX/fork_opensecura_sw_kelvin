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

#include <inttypes.h>
#include <limits.h>

#include "benchmarks/benchmark.h"
#include "hw/top_matcha/sw/autogen/top_matcha.h"
#include "sw/device/lib/base/math.h"
#include "sw/device/lib/dif/dif_ml_top.h"
#include "sw/device/lib/dif/dif_rv_plic.h"
#include "sw/device/lib/runtime/hart.h"
#include "sw/device/lib/runtime/irq.h"
#include "sw/device/lib/runtime/log.h"
#include "sw/device/lib/runtime/print.h"
#include "sw/device/lib/testing/test_framework/check.h"
#include "sw/device/lib/testing/test_framework/ottf_test_config.h"
#include "sw/device/lib/testing/test_framework/status.h"
#include "sw/device/lib/testing/test_framework/test_util.h"

#define STRINGIZE(x) #x
#define STR(x) STRINGIZE(x)

OTTF_DEFINE_TEST_CONFIG();

static dif_rv_plic_t plic_smc;
static dif_uart_t smc_uart;
static dif_ml_top_t ml_top;

volatile bool ml_top_finish_done = false;

void _print64(const char* header, uint64_t number) {
  uint32_t number_low = number & 0xFFFFFFFF;
  uint32_t number_hi = number >> 32;
  LOG_INFO("%s: 0x%08x%08x", header, number_hi, number_low);
}

void ottf_external_isr(void) {
  dif_rv_plic_irq_id_t interrupt_id;
  CHECK_DIF_OK(dif_rv_plic_irq_claim(&plic_smc, kTopMatchaPlicTargetIbex0Smc,
                                     &interrupt_id));

  top_matcha_plic_peripheral_smc_t peripheral_id =
      top_matcha_plic_interrupt_for_peripheral_smc[interrupt_id];
  switch (peripheral_id) {
    case kTopMatchaPlicPeripheralMlTop: {
      switch (interrupt_id) {
        case kTopMatchaPlicIrqIdMlTopFinish:
          ml_top_finish_done = true;
          break;
        default:
          CHECK(false, "Unhandled ML_TOP interrupt");
      }
      CHECK_DIF_OK(dif_ml_top_reset_ctrl_en(&ml_top));
      CHECK_DIF_OK(dif_ml_top_irq_acknowledge_all(&ml_top));
      break;
    }
    default:
      CHECK(false, "Unhandled peripheral! %d", peripheral_id);
  }

  CHECK_DIF_OK(dif_rv_plic_irq_complete(&plic_smc, kTopMatchaPlicTargetIbex0Smc,
                                        interrupt_id));
}


void _ottf_main(void) {
  // Initialize the SMC UART to enable logging for non-DV simulation platforms.
  if (kDeviceType != kDeviceSimDV) {
    init_uart(TOP_MATCHA_SMC_UART_BASE_ADDR, &smc_uart);
  }

  CHECK_DIF_OK(dif_rv_plic_init(
      mmio_region_from_addr(TOP_MATCHA_RV_PLIC_SMC_BASE_ADDR), &plic_smc));
  CHECK_DIF_OK(dif_ml_top_init(
      mmio_region_from_addr(TOP_MATCHA_ML_TOP_CORE_BASE_ADDR), &ml_top));
  CHECK_DIF_OK(dif_ml_top_irq_set_enabled(&ml_top, kDifMlTopIrqFinish,
                                          kDifToggleEnabled));
  CHECK_DIF_OK(dif_rv_plic_irq_set_priority(
      &plic_smc, kTopMatchaPlicIrqIdMlTopFinish, kDifRvPlicMaxPriority));
  CHECK_DIF_OK(dif_rv_plic_irq_set_enabled(
      &plic_smc, kTopMatchaPlicIrqIdMlTopFinish, kTopMatchaPlicTargetIbex0Smc,
      kDifToggleEnabled));
  irq_global_ctrl(true);
  irq_external_ctrl(true);

  LOG_INFO("========== Begin Benchmark (%s) ==========", STR(BENCHMARK_NAME));

  // start kelvin
  ml_top_finish_done = false;
  CHECK_DIF_OK(dif_ml_top_release_ctrl_en(&ml_top));

  // wfi
  while (!ml_top_finish_done) {
    wait_for_interrupt();
  }

  BenchmarkOutputHeader* output_header_ptr =
      (BenchmarkOutputHeader*)((TOP_MATCHA_ML_TOP_DMEM_BASE_ADDR +
                                TOP_MATCHA_RAM_ML_DMEM_SIZE_BYTES) -
                              0x40);

  if (output_header_ptr->return_code) {
    LOG_FATAL("Kelvin returned an error: %d", output_header_ptr->return_code);
  }
  uint32_t iterations = output_header_ptr->iterations;
  uint64_t cycles = output_header_ptr->cycles;
  uint64_t average_cycles = udiv64_slow(cycles, iterations, NULL);
  LOG_INFO("Iterations: %d", iterations);
  _print64("Total Cycles", cycles);
  _print64("Average Cycles per Iteration", average_cycles);
  LOG_INFO("========== End Benchmark ==========");
  while (true) {
    wait_for_interrupt();
  };
}
