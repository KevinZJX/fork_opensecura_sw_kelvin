// Copyright 2023 Google LLC.
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// A starting functions for simple kelvin programs.

/**
 * Entry point.
 */
        .section ._init
        .balign 4
        .global _start
        .type _start, @function
_start:
        ###############################################
        # Put all scalar registers into a known state #
        ###############################################
        la   sp, __stack_end__
        la   gp, _global_pointer
        la   s0, __heap_start__
        sw   s0, _heap_ptr, s1
        mv   tp, zero
        mv   t1, zero
        mv   t2, zero
        mv   s0, zero
        mv   s1, zero
        mv   a1, zero
        mv   a2, zero
        mv   a3, zero
        mv   a4, zero
        mv   a5, zero
        mv   a6, zero
        mv   a7, zero
        mv   s2, zero
        mv   s3, zero
        mv   s4, zero
        mv   s5, zero
        mv   s6, zero
        mv   s7, zero
        mv   s8, zero
        mv   s9, zero
        mv   s10, zero
        mv   s11, zero
        mv   t3, zero
        mv   t4, zero
        mv   t5, zero
        mv   t6, zero

        # (TODO:lundong) uncomment the following
        # when runtime error in iree targets is resolved
        # Zero out the bss section
        # la   a0, __bss_start__
        # la   a1, __bss_end__
        # call crt_section_clear

        # Initialize arrays
        la   s0, __init_array_start__
        la   s1, __init_array_end__
        bgeu s0, s1, init_array_loop_end
init_array_loop:
        lw   t0, 0(s0)
        jalr t0
        addi s0, s0, 0x4
        bltu s0, s1, init_array_loop
init_array_loop_end:


        #############
        # Call main #
        #############
        li   a0, 0  # argv
        li   a1, 0  # argc
        la   ra, main
        jalr ra, ra
        # Store the application's return value at _ret
        la   t0, _ret
        sw   a0, 0(t0)
        mpause      # Kelvin end of program op
1:
        j    1b
