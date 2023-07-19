/* Copyright 2023 Google LLC. */
/* Licensed under the Apache License, Version 2.0, see LICENSE for details. */
/* SPDX-License-Identifier: Apache-2.0 */

/**
 *  A simple linker script for kelvin core.
 *
 * Memory layout:
 *
 *  TCM_ORIGIN          --->   +=====================+
 *                             |                     |
 *                             |       .text         |
 *                             +---------------------|
 *                             |       .crt          |
 *                             +---------------------+
 *                             |      .rodata        |
 *                             +---------------------+
 *                             |    .init_array      |
 *                             +---------------------+
 *                             |       .data         |
 *                             +---------------------+
 *                             |       .bss          |
 *                             +---------------------+
 *                             |       .heap         |
 *                             |  (All unclamied     |
 *                             |       memory)       |
 *                             |                     |
 *  (TCM_END - stack    --->   +---------------------+
 *     - model_output          |       .stack        |
 *     - output_header)        +---------------------+
 *                             |   .model_output     |
 *  output_header (64B) --->   +---------------------+
 *                             |   .output_header    |
 *  TCM_END             --->   +=====================+
 **/

MEMORY {
  TCM(rwx): ORIGIN = 0x00000000, LENGTH = 0x400000
}

STACK_SIZE = DEFINED(STACK_SIZE) ? STACK_SIZE : DEFINED(__stack_size__) ? __stack_size__ : 0x4000;
OUTPUT_SIZE = DEFINED(OUTPUT_SIZE) ? OUTPUT_SIZE : DEFINED(__output_size__) ? __output_size__ : 0x40;
__output_header_size__ = 0x40;

ENTRY(_start)

SECTIONS {
  . = ORIGIN(TCM);
  .text : ALIGN(4) {
    *(._init)
    *(.text)
    *(.text.*)
    . = ALIGN(4);
    __text_end__ = .;
  } > TCM

  .init.array : ALIGN(4) {
    __init_array_start__ = .;
    *(.init_array)
    *(.init_array.*)
    . = ALIGN(4);
    __init_array_end__ = .;
  } > TCM

  .rodata : ALIGN(4) {
    *(.srodata)
    *(.srodata.*)
    *(.rodata)
    *(.rodata.*)
    . = ALIGN(4);
  } > TCM

  .data : ALIGN(4) {
    __data_start__ = .;
    /**
    * This will get loaded into `gp`, and the linker will use that register for
    * accessing data within [-2048,2047] of `__global_pointer$`.
    *
    * This is much cheaper (for small data) than materializing the
    * address and loading from that (which will take one extra instruction).
    */
    _global_pointer = . + 0x800;
    *(.sdata)
    *(.sdata.*)
    *(.data)
    *(.data.*)
    /* Align on 256 width. */
    . = ALIGN(256);
    __data_end__ = .;
  } > TCM

  .bss : ALIGN(4) {
    __bss_start__ = .;
    *(.sbss)
    *(.sbss.*)
    *(.bss)
    *(.bss.*)
    __bss_end__ = .;
  } > TCM

   /* All the unclaimed memory is used in heap */
  .heap (NOLOAD) : {
    . = ALIGN(64);
    __heap_start__ = .;
    . = ORIGIN(TCM) + LENGTH(TCM) - STACK_SIZE - OUTPUT_SIZE - __output_header_size__ - 63;
    . = ALIGN(64);
    __heap_end__ = .;
  } > TCM

  .stack ORIGIN(TCM) + LENGTH(TCM) - STACK_SIZE - OUTPUT_SIZE - __output_header_size__ (NOLOAD) : {
    __stack_start__ = .;
    . += STACK_SIZE;
    . = ALIGN(64);
    __stack_end__ = .;
  } > TCM

  .model_output ORIGIN(TCM) + LENGTH(TCM) - OUTPUT_SIZE - __output_header_size__ (NOLOAD) : {
    . = ALIGN(64);
    __model_output_start__ = .;
    *(.model_output)
    __model_output_end__ = .;
  } > TCM

  .output_header ORIGIN(TCM) + LENGTH(TCM) - __output_header_size__ (NOLOAD) : {
    /*
     * Model output information (return code and output location/length) is
     * always at the end of TCM.
     */
    __output_header_start__ = .;
    _ret = .;
    *(.model_output_header*)
    __output_header_end__ = .;
  } > TCM
}
