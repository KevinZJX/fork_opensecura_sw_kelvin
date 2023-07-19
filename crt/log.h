// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0
//
// Kelvin logging helper header

#ifndef CRT_LOG_H_
#define CRT_LOG_H_

#include <stdio.h>

#define LOG_MAX_SZ 256

static inline void kelvin_simprint(const char *_string) {
  __asm__ volatile("flog %0 \n\t" : : "r"(_string));
}

#define SIMLOG(fmt, ...)                                 \
  do {                                                   \
    char tmp_log_msg[LOG_MAX_SZ];                        \
    snprintf(tmp_log_msg, LOG_MAX_SZ, fmt, __VA_ARGS__); \
    kelvin_simprint(tmp_log_msg);                        \
  } while (0)

#define LOG_ERROR(msg, args...) SIMLOG("%s |" msg "\n", "ERROR", ##args)
#define LOG_WARN(msg, args...) SIMLOG("%s |" msg "\n", "WARN", ##args)
#define LOG_INFO(msg, args...) SIMLOG("%s |" msg "\n", "INFO", ##args)
#define LOG_DEBUG(msg, args...) SIMLOG("%s |" msg "\n", "DEBUG", ##args)
#define LOG_NOISY(msg, args...) SIMLOG("%s |" msg "\n", "NOISY", ##args)

#endif  // CRT_LOG_H_
