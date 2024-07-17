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
#include "sw/device/lib/dif/dif_tlul_mailbox.h"
#include "sw/device/lib/dif/dif_uart.h"
#include "sw/device/lib/runtime/hart.h"
#include "sw/device/lib/runtime/irq.h"
#include "sw/device/lib/spi_flash.h"
#include "sw/device/lib/testing/test_framework/check.h"
#include "sw/device/lib/testing/test_framework/ottf_test_config.h"
#include "sw/device/lib/testing/test_framework/test_util.h"
/* clang-format off */
#include "benchmarks/benchmark.h"
/* clang-format on */

#define STRINGIZE(x) #x
#define STR(x) STRINGIZE(x)

// In order to include the model data generate from Bazel, include the header
// using the name passed as a macro. For some reason this binary (vs Kelvin)
// adds space when concatinating so use the model format -smc_bin.h.
#define SMC_BINARY_DIRECTORY BENCHMARK_PATH
#define SMC_BINARY_TYPE smc_bin.h
#define SMC_BINARY STR(SMC_BINARY_DIRECTORY/BENCHMARK_NAME-SMC_BINARY_TYPE)
#include SMC_BINARY

static dif_gpio_t gpio;
static dif_rv_plic_t plic_sec;
static dif_tlul_mailbox_t tlul_mailbox;
static dif_pinmux_t pinmux;
static dif_smc_ctrl_t smc_ctrl;
static dif_uart_t uart;

OTTF_DEFINE_TEST_CONFIG();

void ottf_external_isr(void) {
  uint32_t rx;
  dif_rv_plic_irq_id_t plic_irq_id;

  CHECK_DIF_OK(dif_rv_plic_irq_claim(&plic_sec, kTopMatchaPlicTargetIbex0,
                                     &plic_irq_id));
  top_matcha_plic_peripheral_t peripheral_id =
      top_matcha_plic_interrupt_for_peripheral[plic_irq_id];

  switch (peripheral_id) {
    case kTopMatchaPlicPeripheralTlulMailboxSec: {
      CHECK_DIF_OK(dif_tlul_mailbox_irq_acknowledge(&tlul_mailbox,
                                                    kDifTlulMailboxIrqRtirq));
      CHECK_DIF_OK(dif_tlul_mailbox_read_message(&tlul_mailbox, &rx));
      uint32_t pin = rx >> 16;
      uint32_t value = rx & 0xFFFF;
      CHECK_DIF_OK(dif_gpio_write(&gpio, pin, value));
      break;
    }
    default:
      LOG_FATAL("Unhandled interrupt");
      break;
  }

  CHECK_DIF_OK(dif_rv_plic_irq_complete(&plic_sec, kTopMatchaPlicTargetIbex0,
                                        plic_irq_id));
}

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

// PinMux: Total inference GPIO J52.5/CS  Sparrow (IOR7) :: PMOD3.7 on Nexus
// (IOD4)
//           Per inference GPIO J52.7/SCK Sparrow (IOR7) :: PMOD3.8 on Nexus
//           (IOD5)
#if defined(MATCHA_SPARROW)
  CHECK_DIF_OK(dif_pinmux_output_select(&pinmux, kTopMatchaPinmuxMioOutIor7,
                                        kTopMatchaPinmuxOutselGpioGpio16));
  CHECK_DIF_OK(dif_pinmux_output_select(&pinmux, kTopMatchaPinmuxMioOutIoa1,
                                        kTopMatchaPinmuxOutselGpioGpio17));
#else
  CHECK_DIF_OK(dif_pinmux_output_select(&pinmux, kTopMatchaPinmuxMioOutIod4,
                                        kTopMatchaPinmuxOutselGpioGpio16));
  CHECK_DIF_OK(dif_pinmux_output_select(&pinmux, kTopMatchaPinmuxMioOutIod5,
                                        kTopMatchaPinmuxOutselGpioGpio17));
#endif
  CHECK_DIF_OK(
      dif_gpio_init(mmio_region_from_addr(TOP_MATCHA_GPIO_BASE_ADDR), &gpio));
  CHECK_DIF_OK(dif_gpio_output_set_enabled(&gpio, ML_RUN_INDICATOR_IO,
                                           kDifToggleEnabled));
  CHECK_DIF_OK(dif_gpio_output_set_enabled(&gpio, ML_TOGGLE_PER_INF_IO,
                                           kDifToggleEnabled));

  LOG_INFO("Loading Kelvin binary");
  spi_flash_init();
  CHECK_DIF_OK(load_file_from_tar(
      "kelvin.bin", (void*)TOP_MATCHA_ML_TOP_DMEM_BASE_ADDR,
      (TOP_MATCHA_ML_TOP_DMEM_BASE_ADDR + TOP_MATCHA_RAM_ML_DMEM_SIZE_BYTES)));

  if (kDeviceType == kDeviceFpgaNexus || kDeviceType == kDeviceAsic) {
    LOG_INFO("Loading SMC binary");
    memcpy((void*)TOP_MATCHA_RAM_SMC_BASE_ADDR, smc_bin, smc_bin_len);
  }

  // Enable Mailbox Interrupt
  CHECK_DIF_OK(dif_tlul_mailbox_init(
      mmio_region_from_addr(TOP_MATCHA_TLUL_MAILBOX_SEC_BASE_ADDR),
      &tlul_mailbox));
  CHECK_DIF_OK(dif_tlul_mailbox_irq_set_enabled(
      &tlul_mailbox, kDifTlulMailboxIrqRtirq, kDifToggleEnabled));

  CHECK_DIF_OK(dif_rv_plic_init(
      mmio_region_from_addr(TOP_MATCHA_RV_PLIC_BASE_ADDR), &plic_sec));
  CHECK_DIF_OK(dif_rv_plic_irq_set_enabled(
      &plic_sec, kTopMatchaPlicIrqIdTlulMailboxSecRtirq,
      kTopMatchaPlicTargetIbex0, kDifToggleEnabled));
  CHECK_DIF_OK(dif_rv_plic_irq_set_priority(
      &plic_sec, kTopMatchaPlicIrqIdTlulMailboxSecRtirq, 1));
  irq_global_ctrl(true);
  irq_external_ctrl(true);

  CHECK_DIF_OK(dif_smc_ctrl_set_en(&smc_ctrl));
  irq_global_ctrl(true);
  irq_external_ctrl(true);

  while (true) {
    wait_for_interrupt();
  }
  __builtin_unreachable();
}
