// Copyright 2023 Google LLC
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// Host tool to generate the intrinsic header and toolchain op files

// Encoding generator "kelvin-opc.[c,h]" and kelvin_intrinsics.h

#include <assert.h>
#include <stdint.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

#include <algorithm>
#include <string>
#include <vector>

FILE* fc_ = nullptr;
FILE* fh_ = nullptr;
FILE* fi_ = nullptr;

enum OpCode {
  kOpNone,
  kOpS,
  kOpST,
  kOpD,
  kOpDS,
  kOpDST,
  kOpVd,
  kOpVdS,
  kOpVdT,
  kOpVdST,
  kOpVdVs,
  kOpVdVsT,
  kOpVdVsVt,
  kOpVdVsTVr,
  kOpVdVsVtVr,
};

constexpr int kPad1 = 24;
constexpr int kPad2 = 56;

constexpr uint32_t kVxMask = 1;
constexpr uint32_t kVxShift = 1;
constexpr uint32_t kVxvMask = 1;
constexpr uint32_t kVxvShift = 2;

constexpr uint32_t kFunc1Shift = 2;
constexpr uint32_t kFunc1Mask = 0x7;
constexpr uint32_t kFunc2Shift = 26;
constexpr uint32_t kFunc2Mask = 0x3f;
constexpr uint32_t kFunc3Shift = 26;
constexpr uint32_t kFunc3Mask = 0x3f;
constexpr uint32_t kFunc4LoShift = 3;
constexpr uint32_t kFunc4LoMask = 0x3;
constexpr uint32_t kFunc4HiShift = (12 - 2);
constexpr uint32_t kFunc4HiMask = 0xc;

constexpr uint32_t kMMask = 0x1;
constexpr uint32_t kMShift = 5;
constexpr uint32_t kSizeMask = 0x3;
constexpr uint32_t kSizeShift = 12;
constexpr uint32_t kVs1Mask = 0x3f;
constexpr uint32_t kVs1Shift = 14;
constexpr uint32_t kVs2Mask = 0x3f;
constexpr uint32_t kVs2Shift = 20;

void Space() {
  for (auto fp : {fc_, fh_, fi_}) {
    fprintf(fp, "\n");
  }
}

// Add same comment to all three output.
void Comment(std::string s) {
  for (auto fp : {fc_, fh_, fi_}) {
    fprintf(fp, "// %s\n", s.c_str());
  }
}

void Header() {
  // OSS header.
  for (auto fp : {fc_, fh_, fi_}) {
    fprintf(fp,
            "// Copyright 2023 Google LLC\n"
            "// Licensed under the Apache License, Version 2.0, see LICENSE "
            "for details.\n"
            "// SPDX-License-Identifier: Apache-2.0\n"
            "//\n"
            "// clang-format off\n\n");
  }
  fprintf(fh_,
          "#ifndef PATCHES_KELVIN_KELVIN_OPC_H_\n"
          "#define PATCHES_KELVIN_KELVIN_OPC_H_\n\n");

  // intrinsic header file header
  fprintf(fi_,
          "// Kelvin instruction intrinsics\n\n"
          "#ifndef CRT_KELVIN_INTRINSICS_H_\n"
          "#define CRT_KELVIN_INTRINSICS_H_\n\n");
}

void Footer() {
  fprintf(fh_, "#endif  // PATCHES_KELVIN_KELVIN_OPC_H_\n");
  fprintf(fi_, "#endif  // CRT_KELVIN_INTRINSICS_H_\n");

  for (auto fp : {fc_, fh_, fi_}) {
    // kelvin-ops.c need to set clang-format on differently.
    std::string close_string = (fp == fc_) ? "\n    " : "";
    fprintf(fp, "%s// clang-format on\n", close_string.c_str());
  }
}

// Generate a 32-bit ISA mask/match bitmap based on the base string
uint32_t VMatchMask(const char* a, bool is_mask = true) {
  const int n = strlen(a);
  int i = 0;
  int p = 0;
  uint32_t v = 0;

  while (p < n && i < 32) {
    const char c = a[p];
    p++;
    if (c == '_') {
      continue;
    }
    if (c == '1') {
      v |= 1u << (31 - i);
    }
    if (is_mask && c == '0') {
      v |= 1u << (31 - i);
    }
    i++;
  }

  if (i != 32) {
    fprintf(stderr, "ERROR: %s\n", a);
    exit(-1);
  }

  return v;
}

uint32_t VMask(const char* a) { return VMatchMask(a, /*is_mask=*/true); }

uint32_t VMatch(const char* a) { return VMatchMask(a, /*is_mask=*/false); }

// Create match bitmap for bit[31...26]
void MmOpField(uint8_t bits, uint32_t& match, uint32_t& mask) {
  match |= (bits & kFunc2Mask) << kFunc2Shift;
}

// Create match bitmap for bit[13...12, 4...3]
void MmOpField2(uint8_t bits, uint32_t& match, uint32_t& mask) {
  match |= (bits & kFunc4LoMask) << kFunc4LoShift;
  match |= (bits & kFunc4HiMask) << kFunc4HiShift;
}

// Create match bitmap for bit[13...12]
void MmSize(uint8_t sz, uint32_t& match, uint32_t& mask) {
  match |= (sz & kSizeMask) << kSizeShift;
}

// Create match bitmap for bit[5]
void MmStripmine(uint32_t& match, uint32_t& mask) {
  match |= kMMask << kMShift;
}

// Create match and mask bitmaps for .vx. bit[1] == 1, and bit[25...20]
// are used to record a scalar register
void MmVx(uint32_t& match, uint32_t& mask) {
  match |= 1 << kVxShift;
  mask |= 1 << (kVs2Shift + 5);  // bit[25] == 0
}

// Create match and mask bitmaps for .vxv, bit[25...20] are used to record a
// scalar register, so bit[25] = 0
void MmVxV(uint32_t& match, uint32_t& mask) {
  match |= 1 << kVxvShift;
  mask |= 1 << (kVs2Shift + 5);  // bit[25] == 0
}

void MmVxVAlternate(uint32_t& match, uint32_t& mask) {
  match |= 1 << (kVs2Shift + 5);  // bit[25] == 1
}

// Create mask bitmap for .xx, both bit[25...20] and bit[19...14]
// are used by scalar registers.
void MmXx(uint32_t& match, uint32_t& mask) {
  mask |= 1 << (kVs1Shift + 0);  // bit[14] == 0
  mask |= 1 << (kVs2Shift + 5);  // bit[25] == 0
}

// Create match and mask bitmaps for bit[19...14] == 0
void MmXs1IsZero(uint32_t& match, uint32_t& mask) {
  match &= ~(kVs1Mask << kVs1Shift);
  mask |= kVs1Mask << kVs1Shift;
}

// Create match and mask bitmaps for bit[25...20] == 0
void MmXs2IsZero(uint32_t& match, uint32_t& mask) {
  match &= ~(kVs2Mask << kVs2Shift);
  mask |= kVs2Mask << kVs2Shift;
}

void Pad(std::string& s, int padding) {
  for (int i = s.length(); i < padding; ++i) {
    s += " ";
  }
}

void TypeHelper(const int type, std::string& op) {
  switch (type) {
    case 0:
      op += ".b";
      break;
    case 1:
      op += ".h";
      break;
    case 2:
      op += ".w";
      break;
  }
}

// Check if the op is within the op_group.
bool CheckVariant(const std::string& op_name,
                  const std::vector<std::string>& op_group) {
  for (auto op : op_group) {
    if (op_name == op) return true;
  }
  return false;
}

// Encode opcode with match, mask, and intrinsic macro entry.
void EncodeCH(std::string name, uint32_t match, uint32_t mask, OpCode type) {
  std::string hname = name;
  std::string iname = name;

  // .h opcode mask/match entry. Capitalized.
  for_each(hname.begin(), hname.end(), [](char& c) {
    c = ::toupper(c);
    if (c == '.') c = '_';
  });

  for_each(iname.begin(), iname.end(), [](char& c) {
    c = ::tolower(c);
    if (c == '.') c = '_';
  });

  // .c opcode entry.
  std::string co;
  co += "{\"" + name + "\",";
  Pad(co, kPad1);
  co += "0, INSN_CLASS_K, \"";
  switch (type) {
    case kOpNone:
      break;
    case kOpS:
      co += "s";
      break;
    case kOpST:
      co += "s,t";
      break;
    case kOpD:
      co += "d";
      break;
    case kOpDS:
      co += "d,s";
      break;
    case kOpDST:
      co += "d,s,t";
      break;
    case kOpVd:
      co += "Vd";
      break;
    case kOpVdS:
      co += "Vd,s";
      break;
    case kOpVdT:
      co += "Vd,t";
      break;
    case kOpVdST:
      co += "Vd,s,t";
      break;
    case kOpVdVs:
      co += "Vd,Vs";
      break;
    case kOpVdVsT:
      co += "Vd,Vs,t";
      break;
    case kOpVdVsVt:
      co += "Vd,Vs,Vt";
      break;
    case kOpVdVsTVr:
      co += "Vd,Vs,t,Vr";
      break;
    case kOpVdVsVtVr:
      co += "Vd,Vs,Vt,Vr";
      break;
    default:
      assert(false && "EncodeCH::kOp");
  }
  co += "\",";
  Pad(co, kPad2);
  co += "MATCH_" + hname + ", MASK_" + hname + ", match_opcode, 0 },";
  // intrinsics
  std::string io;
  io = "#define " + iname;

  // Append .xx to scalar opcodes intrinsic definitions.
  //  eg. avoid std::min/max collisions.
  switch (type) {
    case kOpDS:
      if (iname.find("_x") == std::string::npos) {
        io += "_x";
      }
      break;
    case kOpDST:
      if (iname.find("_x") == std::string::npos) {
        io += "_xx";
      }
      break;
    default:
      break;
  }

  switch (type) {
    case kOpNone:
      io += "()";
      break;
    case kOpS:
      io += "(s)";
      break;
    case kOpST:
      io += "(s, t)";
      break;
    case kOpD:
      io += "(d)";
      break;
    case kOpDS:
      io += "(d, s)";
      break;
    case kOpDST:
      io += "(d, s, t)";
      break;
    case kOpVd:
      io += "(Vd)";
      break;
    case kOpVdS:
      io += "(Vd, s)";
      break;
    case kOpVdT:
      io += "(Vd, t)";
      break;
    case kOpVdST:
      io += "(Vd, s, t)";
      break;
    case kOpVdVs:
      io += "(Vd, Vs)";
      break;
    case kOpVdVsT:
      io += "(Vd, Vs, t)";
      break;
    case kOpVdVsVt:
      io += "(Vd, Vs, Vt)";
      break;
    case kOpVdVsTVr:
      io += "(Vd, Vs, t, Vr)";
      break;
    case kOpVdVsVtVr:
      io += "(Vd, Vs, Vt, Vr)";
      break;
    default:
      assert(false && "EncodeCH::kOp");
  }

  Pad(io, 36);
  io += "__asm__ __volatile__";

  switch (type) {
    case kOpNone:
      io += "(\"" + name + "\");";
      break;
    case kOpS:
      io += "(ARGS_F_A(\"" + name + "\", %0) : : \"r\"(s))";
      break;
    case kOpST:
      io += "(ARGS_F_A(\"" + name + "\", %0, %1) : : \"r\"(s), \"r\"(t))";
      break;
    case kOpD:
      io += "(ARGS_F_A(\"" + name + "\", %0) : \"=r\"(d) : )";
      break;
    case kOpDS:
      io += "(ARGS_F_A_A(\"" + name + "\", %0, %1) : \"=r\"(d) : \"r\"(s))";
      break;
    case kOpDST:
      io += "(ARGS_F_A_A_A(\"" + name +
            "\", %0, %1, %2) : \"=r\"(d) : \"r\"(s), \"r\"(t))";
      break;
    case kOpVd:
      io += "(ARGS_F_A(\"" + name + "\", Vd) : : )";
      break;
    case kOpVdS:
      io += "(ARGS_F_A_A(\"" + name + "\", Vd, %0) : : \"r\"(s))";
      break;
    case kOpVdT:
      io += "(ARGS_F_A_A(\"" + name + "\", Vd, %0) : : \"r\"(t))";
      break;
    case kOpVdST:
      io +=
          "(ARGS_F_A_A_A(\"" + name + "\", Vd, %0, %1) : : \"r\"(s), \"r\"(t))";
      break;
    case kOpVdVs:
      io += "(ARGS_F_A_A(\"" + name + "\", Vd, Vs))";
      break;
    case kOpVdVsT:
      io += "(ARGS_F_A_A_A(\"" + name + "\", Vd, Vs, %0) : : \"r\"(t))";
      break;
    case kOpVdVsVt:
      io += "(ARGS_F_A_A_A(\"" + name + "\", Vd, Vs, Vt))";
      break;
    case kOpVdVsTVr:
      io += "(ARGS_F_A_A_A_A(\"" + name + "\", Vd, Vs, %0, Vr) : : \"r\"(t))";
      break;
    case kOpVdVsVtVr:
      io += "(ARGS_F_A_A_A_A(\"" + name + "\", Vd, Vs, Vt, Vr))";
      break;
    default:
      assert(false && "EncodeCH::kOp");
  }

  // Order these instructions.
  std::vector<std::string> always{
      "ebreak", "ecall", "eexit", "eyield", "ectxsw", "mret", "mpause",
      "flog",   "slog",  "klog",  "clog",   "vld",    "vst",  "vsq"};
  for (auto op : always) {
    if (name.find(op) != std::string::npos) {
      std::string v = "volatile";
      size_t pos = io.find(v) + v.length();
      io.insert(pos, "_always");
    }
  }

  // Update load/store.
  std::vector<std::string> ldst{"vld", "vst"};
  for (auto op : ldst) {
    if (name.find(op) != std::string::npos) {
      // Update base.
      std::string p = "p_x";
      auto pos = io.find(p);
      if (pos != std::string::npos) {
        std::string r = ": \"r\"(s)";
        int pos = io.find(r);
        io.insert(pos, "\"=r\"(s) ");

        pos = io.find(": \"r\"(s), \"r\"(t)");
        if (pos > 0) {
          // : "r"(s), "r"(t)  # from
          // : "r"(t), "0"(s)  # to
          io[pos + 6] = 't';
          io[pos + 11] = '0';
          io[pos + 14] = 's';
        } else {
          // : "r"(s)  # from
          // : "0"(s)  # to
          pos = io.find(": \"r\"(s)");
          io[pos + 3] = '0';
        }
      }
      // Memory side-effects.
      pos = io.length() - 1;
      io.insert(pos, " : \"memory\"");
    }
  }

  // file write
  Pad(hname, 20);

  fprintf(fc_, "%s\n", co.c_str());

  fprintf(fh_, "#define MATCH_%s 0x%08x\n", hname.c_str(), match);
  fprintf(fh_, "#define MASK_%s  0x%08x\n", hname.c_str(), mask);

  fprintf(fi_, "%s\n", io.c_str());
}

void Encode(std::string name, std::string op) {
  EncodeCH(name, VMatch(op.c_str()), VMask(op.c_str()), kOpNone);
}

void EncodeS(std::string name, std::string op) {
  EncodeCH(name, VMatch(op.c_str()), VMask(op.c_str()), kOpS);
}

void EncodeGetVl() {
  const char* base = "00010_00_xxxxx_xxxxx_000_xxxxx_11101_11";
  const uint32_t vmatch_base = VMatch(base);
  const uint32_t vmask_base = VMask(base);

  for (auto m : {false, true}) {
    for (int srcs = 0; srcs < 3; ++srcs) {
      for (int sz = 0; sz < 3; ++sz) {
        std::string op = "get";
        uint32_t vmatch = vmatch_base;
        uint32_t vmask = vmask_base;
        OpCode opcode;
        if (srcs == 0) {
          op += "maxvl";
        } else {
          op += "vl";
        }
        TypeHelper(sz, op);
        switch (srcs) {
          case 0: {
            opcode = kOpD;
            MmXs1IsZero(vmatch, vmask);
            MmXs2IsZero(vmatch, vmask);
          } break;
          case 1: {
            opcode = kOpDS;
            MmXs2IsZero(vmatch, vmask);
            op += ".x";
          } break;
          case 2: {
            opcode = kOpDST;
            op += ".xx";
          } break;
          default:
            break;
        }
        MmXx(vmatch, vmask);
        // Set size match bitmap.
        constexpr uint32_t kSystemSzShift = 25;
        vmatch |= (sz & kSizeMask) << kSystemSzShift;
        if (m) {
          op += ".m";
          constexpr uint32_t kSystemMShift = 27;
          vmatch |= (kMMask << kSystemMShift);
        }

        EncodeCH(op.c_str(), vmatch, vmask, opcode);
      }
    }
    // Encoded files have spaces between the normal mode and the stripmine mode.
    if (!m) Space();
  }
}

void EncodeVLdSt(std::string name, const int index) {
  const char* base = "000000_xxxxxx_xxxxxx_00_xxxxxx_0_111_11";
  const uint32_t vmatch_base = VMatch(base);
  const uint32_t vmask_base = VMask(base);

  bool is_stq = (name == "vstq");

  for (int sz = 0; sz < 3; ++sz) {
    for (auto m : {false, true}) {
      for (int mode = 0; mode < 8; ++mode) {
        for (auto xs2_is_zero : {false, true}) {
          if (mode != 0 && xs2_is_zero) continue;
          // Can't set s and l without p (no .ls mode)
          if (mode == 3) continue;
          // stq only has .s and .sp mode
          if (mode != 2 && mode != 6 && is_stq) continue;

          std::string op = name;
          uint32_t vmatch = vmatch_base;
          uint32_t vmask = vmask_base;

          TypeHelper(sz, op);
          MmSize(sz, vmatch, vmask);

          switch (mode) {
            case 0:
              if (xs2_is_zero) op += ".p";
              break;
            case 1:
              op += ".l";
              break;
            case 2:
              op += ".s";
              break;
            case 4:
              op += ".p";
              break;
            case 5:
              op += ".lp";
              break;
            case 6:
              op += ".sp";
              break;
            case 7:
              op += ".tp";
              break;
            default:
              break;
          }
          MmOpField(index | mode | (xs2_is_zero ? 0b100 : 0), vmatch, vmask);

          op += mode ? ".xx" : ".x";
          if (mode == 0) {
            MmXs2IsZero(vmatch, vmask);
          }
          MmXx(vmatch, vmask);

          if (m) {
            op += ".m";
            MmStripmine(vmatch, vmask);
          }

          EncodeCH(op.c_str(), vmatch, vmask, mode ? kOpVdST : kOpVdS);
        }
      }
    }
  }
}

void EncodeVDup(std::string name, const int index) {
  const char* base = "000000_xxxxxx_xxxxxx_00_xxxxxx_0_111_11";
  const uint32_t vmatch_base = VMatch(base);
  const uint32_t vmask_base = VMask(base);

  for (int sz = 0; sz < 3; ++sz) {
    for (auto m : {false, true}) {
      std::string op = name;
      uint32_t vmatch = vmatch_base;
      uint32_t vmask = vmask_base;

      TypeHelper(sz, op);
      MmSize(sz, vmatch, vmask);

      MmOpField(index, vmatch, vmask);

      op += ".x";
      MmXs1IsZero(vmatch, vmask);
      MmXx(vmatch, vmask);

      if (m) {
        op += ".m";
        MmStripmine(vmatch, vmask);
      }

      EncodeCH(op.c_str(), vmatch, vmask, kOpVdT);
    }
  }
}

void EncodeVCget(std::string name, const int index) {
  const char* base = "000000_xxxxxx_xxxxxx_00_xxxxxx_0_111_11";
  const uint32_t vmatch_base = VMatch(base);
  const uint32_t vmask_base = VMask(base);

  const int sz = 2;

  std::string op = name;
  uint32_t vmatch = vmatch_base;
  uint32_t vmask = vmask_base;

  MmSize(sz, vmatch, vmask);
  MmOpField(index, vmatch, vmask);
  MmXs1IsZero(vmatch, vmask);
  MmXs2IsZero(vmatch, vmask);

  EncodeCH(op.c_str(), vmatch, vmask, kOpVd);
}

void Encode000(std::string name, const int index) {
  const char* base = "000000_xxxxxx_xxxxxx_00_xxxxxx_0_000_00";
  const uint32_t vmatch_base = VMatch(base);
  const uint32_t vmask_base = VMask(base);

  bool has_u =
      CheckVariant(name, {"vlt", "vle", "vgt", "vge", "vabsd", "vmax", "vmin"});

  bool no_vv = (name == "vrsub");

  for (int sz = 0; sz < 3; ++sz) {
    for (auto m : {false, true}) {
      for (auto u : {false, true}) {
        if (u && !has_u) continue;
        // The group has .{u}.{vv, vx}.{m} variations.
        for (auto x : {false, true}) {
          if (!x && no_vv) continue;

          std::string op = name;
          uint32_t vmatch = vmatch_base;
          uint32_t vmask = vmask_base;

          TypeHelper(sz, op);
          MmSize(sz, vmatch, vmask);

          if (u) {
            op += ".u";
          }
          MmOpField(index | u, vmatch, vmask);

          op += x ? ".vx" : ".vv";
          if (x) {
            MmVx(vmatch, vmask);
          }

          if (m) {
            op += ".m";
            MmStripmine(vmatch, vmask);
          }

          EncodeCH(op.c_str(), vmatch, vmask, x ? kOpVdVsT : kOpVdVsVt);
        }
      }
    }
  }
}

void Encode100(std::string name, const int index) {
  const char* base = "000000_xxxxxx_xxxxxx_00_xxxxxx_0_100_00";
  const uint32_t vmatch_base = VMatch(base);
  const uint32_t vmask_base = VMask(base);

  bool has_v_only = CheckVariant(name, {"vpadd", "vpsub"});

  bool no_sz_0 =
      CheckVariant(name, {"vaddw", "vsubw", "vacc", "vpadd", "vpsub"});

  bool has_r = CheckVariant(name, {"vhadd", "vhsub"});

  for (int sz = 0; sz < 3; ++sz) {
    if (sz == 0 && no_sz_0) continue;
    // This group has [.r, .u, .ur].[v, vv, vx].{m} variations.
    for (auto m : {false, true}) {
      for (auto u : {false, true}) {
        for (auto r : {false, true}) {
          if (r && !has_r) continue;
          for (auto v : {false, true}) {
            // Skip this config for the match encoding.
            if (v && !has_v_only) continue;
            bool is_v_only = !v && has_v_only;
            for (auto is_vx : {false, true}) {
              // Skip this to sort .v before .vv and .vx
              if (!v && !is_vx && has_v_only) continue;

              std::string op = name;
              uint32_t vmatch = vmatch_base;
              uint32_t vmask = vmask_base;

              TypeHelper(sz, op);
              MmSize(sz, vmatch, vmask);

              if (u) {
                op += ".u";
              }

              if (has_r) {
                op += (r && !u) ? ".r" : (r ? "r" : "");
              }

              MmOpField(index | (u ? 0b1 : 0) | (has_r && r ? 0b10 : 0), vmatch,
                        vmask);

              op += is_v_only ? ".v" : is_vx ? ".vx" : ".vv";
              if (is_v_only) {
                MmXs2IsZero(vmatch, vmask);
              }
              if (is_vx) {
                MmVx(vmatch, vmask);
              }

              if (m) {
                op += ".m";
                MmStripmine(vmatch, vmask);
              }

              EncodeCH(op.c_str(), vmatch, vmask,
                       is_v_only ? kOpVdVs
                       : is_vx   ? kOpVdVsT
                                 : kOpVdVsVt);
            }
          }
        }
      }
    }
  }
}

void Encode001(std::string name, const int index) {
  const char* base = "000000_xxxxxx_xxxxxx_00_xxxxxx_0_001_00";
  const uint32_t vmatch_base = VMatch(base);
  const uint32_t vmask_base = VMask(base);

  bool is_typeless =
      CheckVariant(name, {"vnot", "vmv", "acset", "actr", "adwinit"});

  bool is_typeless_vv = CheckVariant(name, {"vand", "vor", "vxor", "vmvp"});

  bool is_v_only = CheckVariant(name, {"vnot", "vclb", "vclz", "vcpop", "vmv",
                                       "acset", "actr", "adwinit"});

  bool no_m = CheckVariant(name, {"acset", "actr", "adwinit"});

  for (int sz = 0; sz < 3; ++sz) {
    if (is_typeless && sz != 0) continue;
    // The group has .{v, vv, vx}.{m} variations.
    for (auto m : {false, true}) {
      if (m && no_m) continue;
      for (auto x : {false, true}) {
        // Skip this for the match encoding.
        if (!x && is_v_only) continue;
        // Skip this for the match encoding.
        if (!x && is_typeless_vv && sz != 0) continue;

        std::string op = name;
        uint32_t vmatch = vmatch_base;
        uint32_t vmask = vmask_base;

        if (!is_typeless && !(is_typeless_vv && !x)) {
          TypeHelper(sz, op);
        }
        MmSize(sz, vmatch, vmask);

        MmOpField(index, vmatch, vmask);

        op += is_v_only ? ".v" : x ? ".vx" : ".vv";
        if (is_v_only) {
          MmXs2IsZero(vmatch, vmask);
        }
        if (x) {
          MmVx(vmatch, vmask);
        }

        if (m) {
          op += ".m";
          MmStripmine(vmatch, vmask);
        }

        EncodeCH(op.c_str(), vmatch, vmask,
                 is_v_only ? kOpVdVs
                 : x       ? kOpVdVsT
                           : kOpVdVsVt);
      }
    }
  }
}

void Encode010(std::string name, const int index) {
  const char* base = "000000_xxxxxx_xxxxxx_00_xxxxxx_0_010_00";
  const uint32_t vmatch_base = VMatch(base);
  const uint32_t vmask_base = VMask(base);

  bool has_r = CheckVariant(
      name, {"vsha", "vshl", "vsrans", "vsransu", "vsraqs", "vsraqsu"});

  bool no_16bit = CheckVariant(name, {"vsraqs", "vsraqsu"});

  bool no_32bit =
      CheckVariant(name, {"vsrans", "vsransu", "vsraqs", "vsraqsu"});

  for (int sz = 0; sz < 3; ++sz) {
    if (no_16bit && sz == 1) continue;
    if (no_32bit && sz == 2) continue;
    // The group has .{r}.{vv, vx}.{m} variants
    for (auto m : {false, true}) {
      for (auto r : {false, true}) {
        if (r && !has_r) continue;
        for (auto x : {false, true}) {
          std::string op = name;
          uint32_t vmatch = vmatch_base;
          uint32_t vmask = vmask_base;

          TypeHelper(sz, op);
          MmSize(sz, vmatch, vmask);

          if (r) {
            op += ".r";
          }
          MmOpField(index | (r ? 0b10 : 0), vmatch, vmask);

          op += x ? ".vx" : ".vv";
          if (x) {
            MmVx(vmatch, vmask);
          }

          if (m) {
            op += ".m";
            MmStripmine(vmatch, vmask);
          }

          EncodeCH(op.c_str(), vmatch, vmask, x ? kOpVdVsT : kOpVdVsVt);
        }
      }
    }
  }
}

void Encode011(std::string name, const int index) {
  const char* base = "000000_xxxxxx_xxxxxx_00_xxxxxx_0_011_00";
  const uint32_t vmatch_base = VMatch(base);
  const uint32_t vmask_base = VMask(base);

  bool has_u = CheckVariant(name, {"vmuls", "vmulw"});
  bool has_r = CheckVariant(name, {"vmulh", "vmulhu", "vdmulh"});
  bool has_n = (name == "vdmulh");
  assert(!(has_n && !has_r));

  for (int sz = 0; sz < 3; ++sz) {
    // The group support .{u, r, rn}.{vx, vv},{m} variant
    for (auto m : {false, true}) {
      for (auto u : {false, true}) {
        if (u && !has_u) continue;
        for (auto r : {false, true}) {
          if (r && !has_r) continue;
          for (auto n : {false, true}) {
            if (n && !(has_n && r)) continue;
            for (auto x : {false, true}) {
              std::string op = name;
              uint32_t vmatch = vmatch_base;
              uint32_t vmask = vmask_base;

              TypeHelper(sz, op);
              MmSize(sz, vmatch, vmask);

              if (u) {
                op += ".u";
              }
              if (r) {
                op += ".r";
                if (n) {
                  op += "n";
                }
              }
              MmOpField(index | u | n | (r ? 0b10 : 0), vmatch, vmask);

              op += x ? ".vx" : ".vv";
              if (x) {
                MmVx(vmatch, vmask);
              }

              if (m) {
                op += ".m";
                MmStripmine(vmatch, vmask);
              }

              EncodeCH(op.c_str(), vmatch, vmask, x ? kOpVdVsT : kOpVdVsVt);
            }
          }
        }
      }
    }
  }
}

void Encode110(std::string name, const int index) {
  const char* base = "000000_xxxxxx_xxxxxx_00_xxxxxx_0_110_00";
  const uint32_t vmatch_base = VMatch(base);
  const uint32_t vmask_base = VMask(base);

  std::vector<std::string> slide_group = {"vsliden", "vslidehn", "vslidevn",
                                          "vslidep", "vslidehp", "vslidevp"};

  bool has_range = CheckVariant(name, slide_group);
  int range = has_range ? 3 : 0;
  bool is_m0 = CheckVariant(name, {"vsliden", "vslidep"});
  bool is_m1 =
      CheckVariant(name, {"vslidehn", "vslidehp", "vslidevn", "vslidevp"});

  for (int sz = 0; sz < 3; ++sz) {
    for (auto m : {false, true}) {
      if (!m && is_m1) continue;
      if (m && is_m0) continue;
      for (int n = 0; n <= range; ++n) {
        for (auto x : {false, true}) {
          std::string op = name;
          uint32_t vmatch = vmatch_base;
          uint32_t vmask = vmask_base;

          TypeHelper(sz, op);
          MmSize(sz, vmatch, vmask);

          MmOpField(index | n, vmatch, vmask);

          if (has_range) {
            op += "." + std::to_string(n + 1);
          }

          op += x ? ".vx" : ".vv";
          if (x) {
            MmVx(vmatch, vmask);
          }

          if (m) {
            op += ".m";
            MmStripmine(vmatch, vmask);
          }

          EncodeCH(op.c_str(), vmatch, vmask, x ? kOpVdVsT : kOpVdVsVt);
        }
      }
    }
  }
}

void EncodeVvv(std::string name, const int index, const bool alt = false) {
  const char* base = "xxxxxx_xxxxxx_xxxxxx_00_xxxxxx_0_00_001";
  const uint32_t vmatch_base = VMatch(base);
  const uint32_t vmask_base = VMask(base);

  std::string op = name;
  uint32_t vmatch = vmatch_base;
  uint32_t vmask = vmask_base;

  MmOpField2(index, vmatch, vmask);

  op += ".vxv";
  MmVxV(vmatch, vmask);

  if (alt) {
    MmVxVAlternate(vmatch, vmask);
  }

  EncodeCH(op.c_str(), vmatch, vmask, kOpVdVsTVr);
}

int main() {
  fc_ = fopen("kelvin-opc.c", "wt");
  fh_ = fopen("kelvin-opc.h", "wt");
  fi_ = fopen("kelvin_intrinsics.h", "wt");
  assert(fc_);
  assert(fh_);
  assert(fi_);

  Header();

  Encode("eexit", "00000_01_00000_00000_000_00000_11100_11");
  Encode("eyield", "00000_10_00000_00000_000_00000_11100_11");
  Encode("ectxsw", "00000_11_00000_00000_000_00000_11100_11");
  Encode("mpause", "00001_00_00000_00000_000_00000_11100_11");
  Space();
  EncodeS("flog", "01111_00_00000_xxxxx_000_00000_11101_11");
  EncodeS("slog", "01111_00_00000_xxxxx_001_00000_11101_11");
  EncodeS("clog", "01111_00_00000_xxxxx_010_00000_11101_11");
  EncodeS("klog", "01111_00_00000_xxxxx_011_00000_11101_11");
  Space();
  Encode("flushall", "00100_11_00000_00000_000_00000_11101_11");
  EncodeS("flushat", "00100_11_00000_xxxxx_000_00000_11101_11");
  Space();

  EncodeGetVl();
  Space();

  Comment("111 Load/Store");
  EncodeVLdSt("vld", 0);
  EncodeVLdSt("vst", 8);
  EncodeVDup("vdup", 16);
  EncodeVCget("vcget", 20);
  EncodeVLdSt("vstq", 26);

  Space();
  Comment("000 Arithmetic");
  Encode000("vadd", 0);
  Encode000("vsub", 1);
  Encode000("vrsub", 2);
  Encode000("veq", 6);
  Encode000("vne", 7);
  Encode000("vlt", 8);
  Encode000("vle", 10);
  Encode000("vgt", 12);
  Encode000("vge", 14);
  Encode000("vabsd", 16);
  Encode000("vmax", 18);
  Encode000("vmin", 20);
  Encode000("vadd3", 24);

  Space();
  Comment("100 Arithmetic2");
  Encode100("vadds", 0);
  Encode100("vsubs", 2);
  Encode100("vaddw", 4);
  Encode100("vsubw", 6);
  Encode100("vacc", 10);
  Encode100("vpadd", 12);
  Encode100("vpsub", 14);
  Encode100("vhadd", 16);
  Encode100("vhsub", 20);

  Space();
  Comment("001 Logical");
  Encode001("vand", 0);
  Encode001("vor", 1);
  Encode001("vxor", 2);
  Encode001("vnot", 3);
  Encode001("vrev", 4);
  Encode001("vror", 5);
  Encode001("vclb", 8);
  Encode001("vclz", 9);
  Encode001("vcpop", 10);
  Encode001("vmv", 12);
  Encode001("vmvp", 13);
  Encode001("acset", 16);
  Encode001("actr", 17);
  Encode001("adwinit", 18);

  Space();
  Comment("010 Shift");
  Encode010("vsll", 1);
  Encode010("vsra", 2);
  Encode010("vsrl", 3);
  Encode010("vsha", 8);
  Encode010("vshl", 9);
  Encode010("vsrans", 16);
  Encode010("vsransu", 17);
  Encode010("vsraqs", 24);
  Encode010("vsraqsu", 25);

  Space();
  Comment("011 Mul/Div");
  Encode011("vmul", 0);
  Encode011("vmuls", 2);
  Encode011("vmulw", 4);
  Encode011("vmulh", 8);
  Encode011("vmulhu", 9);
  Encode011("vdmulh", 16);
  Encode011("vmacc", 20);
  Encode011("vmadd", 21);

  Space();
  Comment("110 Shuffle");
  Encode110("vsliden", 0);
  Encode110("vslidevn", 0);
  Encode110("vslidehn", 4);
  Encode110("vslidep", 8);
  Encode110("vslidevp", 8);
  Encode110("vslidehp", 12);
  Encode110("vsel", 16);
  Encode110("vevn", 24);
  Encode110("vodd", 25);
  Encode110("vevnodd", 26);
  Encode110("vzip", 28);

  Space();
  Comment("3arg");
  EncodeVvv("aconv", 8, true);
  EncodeVvv("adwconv", 10, true);
  EncodeVvv("vdwconv", 10);

  Footer();

  fclose(fc_);
  fclose(fh_);
  fclose(fi_);
  fc_ = nullptr;
  fh_ = nullptr;
  fi_ = nullptr;

  return 0;
}
