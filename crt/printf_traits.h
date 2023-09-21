// Copyright 2023 Google LLC
// Licensed under the Apache License, Version 2.0, see LICENSE for details.
// SPDX-License-Identifier: Apache-2.0

#ifndef CRT_PRINTF_TRAITS_H_
#define CRT_PRINTF_TRAITS_H_

#include <cstdint>

// Intentionally left empty, so that
// types without an implementation will cause
// a compile failure.
template <typename T>
struct PrintfTraits {
};

template <>
struct PrintfTraits<int8_t> {
  static constexpr const char* kFmt = "%d";
  static constexpr const char* kFmtHex = "0x%hhx";
};

template <>
struct PrintfTraits<int16_t> {
  static constexpr const char* kFmt = "%hd";
  static constexpr const char* kFmtHex = "0x%hx";
};

template <>
struct PrintfTraits<int32_t> {
  static constexpr const char* kFmt = "%ld";
  static constexpr const char* kFmtHex = "0x%lx";
};

template <>
struct PrintfTraits<uint8_t> {
  static constexpr const char* kFmt = "%u";
  static constexpr const char* kFmtHex = "0x%hhx";
};

template <>
struct PrintfTraits<uint16_t> {
  static constexpr const char* kFmt = "%hu";
  static constexpr const char* kFmtHex = "0x%hx";
};

template <>
struct PrintfTraits<uint32_t> {
  static constexpr const char* kFmt = "%lu";
  static constexpr const char* kFmtHex = "0x%lx";
};

#endif  // CRT_PRINTF_TRAITS_H_
