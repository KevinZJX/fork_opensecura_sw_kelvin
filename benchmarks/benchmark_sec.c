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

#include "hw/top_matcha/sw/autogen/top_matcha.h"
#include "sw/device/lib/arch/device.h"
#include "sw/device/lib/dif/dif_gpio.h"
#include "sw/device/lib/dif/dif_pinmux.h"
#include "sw/device/lib/dif/dif_rv_plic.h"
#include "sw/device/lib/dif/dif_smc_ctrl.h"
#include "sw/device/lib/dif/dif_uart.h"
#include "sw/device/lib/runtime/hart.h"
#include "sw/device/lib/runtime/irq.h"
#include "sw/device/lib/spi_flash.h"
#include "sw/device/lib/testing/test_framework/check.h"
#include "sw/device/lib/testing/test_framework/ottf_test_config.h"
#include "sw/device/lib/testing/test_framework/test_util.h"

#define STRINGIZE(x) #x
#define STR(x) STRINGIZE(x)

// In order to include the model data generate from Bazel, include the header
// using the name passed as a macro. For some reason this binary (vs Kelvin)
// adds space when concatinating so use the model format -smc_bin.h.
#define MODEL_HEADER_DIRECTORY benchmarks
#define MODEL_HEADER_TYPE smc_bin.h
#define MODEL_HEADER STR(MODEL_HEADER_DIRECTORY/BENCHMARK_NAME-MODEL_HEADER_TYPE)
#include MODEL_HEADER

static dif_pinmux_t pinmux;
static dif_smc_ctrl_t smc_ctrl;
static dif_uart_t uart;

OTTF_DEFINE_TEST_CONFIG();

void _ottf_main(void) {
  // Initialize the UART to enable logging for non-DV simulation platforms.
  if (kDeviceType != kDeviceSimDV) {
    init_uart(TOP_MATCHA_UART0_BASE_ADDR, &uart);
  }
  LOG_INFO("Benchmark Main (SEC)");
  CHECK_DIF_OK(dif_pinmux_init(
      mmio_region_from_addr(TOP_MATCHA_PINMUX_AON_BASE_ADDR), &pinmux));
  CHECK_DIF_OK(dif_smc_ctrl_init(
      mmio_region_from_addr(TOP_MATCHA_SMC_CTRL_BASE_ADDR), &smc_ctrl));

  LOG_INFO("Loading Kelvin binary");
  spi_flash_init();
  CHECK_DIF_OK(load_file_from_tar(
      "kelvin.bin", (void*)TOP_MATCHA_ML_TOP_DMEM_BASE_ADDR,
      (TOP_MATCHA_ML_TOP_DMEM_BASE_ADDR + TOP_MATCHA_RAM_ML_DMEM_SIZE_BYTES)));

  if (kDeviceType == kDeviceFpgaNexus || kDeviceType == kDeviceAsic) {
    LOG_INFO("Loading SMC binary");
    memcpy((void*)TOP_MATCHA_RAM_SMC_BASE_ADDR, smc_bin, smc_bin_len);
  }
  CHECK_DIF_OK(dif_smc_ctrl_set_en(&smc_ctrl));
  irq_global_ctrl(true);
  irq_external_ctrl(true);

  while (true) {
    wait_for_interrupt();
  }
  __builtin_unreachable();
}
