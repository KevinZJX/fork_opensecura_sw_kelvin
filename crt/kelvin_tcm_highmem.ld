/* Copyright 2024 Google LLC. */
/* Licensed under the Apache License, Version 2.0, see LICENSE for details. */
/* SPDX-License-Identifier: Apache-2.0 */

MEMORY {
    ITCM(rx): ORIGIN = 0x00000000, LENGTH = 1024K
    DTCM(rw): ORIGIN = 0x00100000, LENGTH = 1024K
}

STACK_SIZE = DEFINED(__stack_size__) ? __stack_size__ : 0x1000;
HEAP_SIZE = DEFINED(__heap_size__) ? __heap_size__ : 0x1000;

ENTRY(_start)

SECTIONS {
    /* ITCM data here */
    . = ORIGIN(ITCM);
    .text : ALIGN(16) {
        *(._init)
        *(.text)
        *(.text.*)
        . = ALIGN(16);
    } > ITCM

    .init.array : ALIGN(16) {
      __init_array_start = .;
      __init_array_start__ = .;
      *(.init_array)
      *(.init_array.*)
      . = ALIGN(16);
      __init_array_end = .;
      __init_array_end__ = .;
    } > ITCM

    .rodata : ALIGN(16) {
      *(.srodata)
      *(.srodata.*)
      *(.rodata)
      *(.rodata.*)
      . = ALIGN(16);
    } > ITCM

    .data : ALIGN(16) {
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
      /**
       * Memory location for the return value from main,
       * which could be inspected by another core in the system.
       **/
      _ret = .;
      . = ALIGN(16);
        __data_end__ = .;
    } > DTCM

    /* DTCM data here */
    . = ORIGIN(DTCM);
    .bss : ALIGN(16) {
      __bss_start__ = .;
      *(.sbss)
      *(.sbss.*)
      *(.bss)
      *(.bss.*)
      __bss_end__ = .;
    } > DTCM

    .heap : ALIGN(16) {
      __heap_start__ = .;
      . += HEAP_SIZE;
      __heap_end__ = .;
    } > DTCM

    .stack : ALIGN(16) {
      __stack_start__ = .;
      . += STACK_SIZE;
      __stack_end__ = .;
    } > DTCM
}