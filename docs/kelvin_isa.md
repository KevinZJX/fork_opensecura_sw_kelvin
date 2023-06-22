# Kelvin Instruction Reference

An ML+SIMD+Scalar instruction set for ML accelerator cores.

[TOC]

## SIMD register configuration

Kelvin has 64 vector registers, `v0` to `v63`, with the vector length of 256-bit
for each of the registers. The register can store data in the format of 8b, 16b,
and 32b, as encoded in the instructions (See the next section for detail).

Kelvin also supports the stripmine behaviors, which utilizes 16 vector registers
with each one 4x the size of the typical register (Also see the details in the
next section).

## SIMD Instructions

The SIMD instructions utilize a register file with 64 entries which serves both
standard arithmetic and logical operations and the domain compute. SIMD lane
size, scalar broadcast, arithmetic operation sign, and stripmine behaviors are
encoded explictly in the opcodes.

The SIMD instructions replace the encoding space of the compressed instruction
set extension (those with 2-bit prefixes 00, 01, and 10). See [The RISC-V
Instruction Set Manual v2.2 "Available 30-bit instruction encoding
spaces"](https://riscv.org/wp-content/uploads/2017/05/riscv-spec-v2.2.pdf) for
quadrupling the available encoding space within the 32-bit format.

### Instruction Encodings

31..26 | 25..20 | 19..14 | 13..12 | 11..6 | 5   | 4..2  | 1..0 | form
:----: | :----: | :----: | :----: | :---: | :-: | :---: | :--: | :--:
func2  | vs2    | vs1    | sz     | vd    | m   | func1 | 00   | .vv
func2  | xs2    | vs1    | sz     | vd    | m   | func1 | 10   | .vx
func2  | 000000 | vs1    | sz     | vd    | m   | func1 | 10   | .v
func2  | xs2    | xs1    | sz     | vd    | m   | 111   | 11   | .xx
func2  | 000000 | xs1    | sz     | vd    | m   | 111   | 11   | .x

<br>

31..26 | 25..20 | 19..14 | 13..12 | 11..6 | 5   | 4..3  | 2..0 | form
:----: | :----: | :----: | :----: | :---: | :-: | :---: | :--: | :--:
vs3    | vs2    | vs1    | func3  | vd    | m   | func3 | 001  | .vvv
vs3    | z,xs2  | vs1    | func3  | vd    | m   | func3 | 101  | .vxv

### Types ".b" ".h" ".w"

The SIMD lane size is encoded in the opcode definition indicating the
destination type. For many opcodes source and destination sizes are the same,
differing for widening and narrowing operations.

op[13:12] | sz   | type
:-------: | :--: | :--:
00        | ".b" | 8b
01        | ".h" | 16b
10        | ".w" | 32b

### Scalar ".vx"

Instructions may use a scalar register to perform a value broadcast (8b, 16b,
32b) to all SIMD lanes of one operand.

op[2:0] | form
:-----: | :------------:
x00     | ".vv"
x10     | ".vx"
x10     | ".v" (xs2==x0)
x11     | ".xx"
x11     | ".x" (xs2==x0)
001     | ".vvv"
101     | ".vxv"

### Signed/Unsigned ".u"

Instructions which may be marked with ".u" have signed and unsigned variants.
See comparisons, arithmetic operations and saturation for usage, the side
effects being typical behaviors unless otherwise noted.

### Stripmine ".m"

The stripmine functionality is an instruction compression mechanism. Frontend
dispatch captures a single instruction, while the backend issue expands to four
operations. Conceptually the register file is reduced from 64 locations to 16,
where a stripmine register must use a mod4 base aligned register (eg. v0, v4,
v8, ...). Normal instruction and stripmine variants may be mixed together.

When stripmining is used in conjunction with instructions which use a register
index as a base to several registers, the offset of +4 (instead of +1) shall be
used. e.g., {vm0,vm1} becomes {{v0,v1,v2,v3},{v4,v5,v6,v7}}.

A machine may elect to distribute a stripmined instruction across multiple ALUs.

op[5] | m
:---: | :--:
0     | ""
1     | ".m"

### 2-arg .xx

Instruction | func2     | Notes
:---------: | :-------: | :--------:
vld         | 00 xx0PSL | 1-arg
vld.l       | 01 xx0PSL |
vld.s       | 02 xx0PSL |
vld.p       | 04 xx0PSL | 1 or 2-arg
vld.lp      | 05 xx0PSL |
vld.sp      | 06 xx0PSL |
vld.tp      | 07 xx0PSL |
vst         | 08 xx1PSL | 1-arg
vst.l       | 09 xx1PSL |
vst.s       | 10 xx1PSL |
vst.p       | 12 xx1PSL | 1 or 2-arg
vst.lp      | 13 xx1PSL |
vst.sp      | 14 xx1PSL |
vst.tp      | 15 xx1PSL |
vdup.x      | 16 x10000 |
vcget       | 20 x10100 | 0-arg
vstq.s      | 26 x11PSL |
vstq.sp     | 30 x11PSL |

To saving encoding space, use the compile time knowledge that if vld.p.xx or
vst.p.xx post-incremented by a zero amount, do not encode x0, instead disable
the post-increment operation so as to reuse the encoding where xs2==x0 for
vld.p.x or vst.p.x which have different base update behavior. If the
post-increment were programmatic behavior then a register where xs2!=x0 would be
used.

### 1-arg .x

Instructions of the format "op.xx vd, xs1, x0" (xs2=x0, the scalar zero
register) are reduced to the shortened form "op.x vd, xs1".

### 0-arg

Instructions of the format "op.xx vd, x0, x0" (xs1=x0, xs2=x0, the scalar zero
register) are reduced to the shortened form "op vd".

### 1-arg .v

Single argument vector operations ".v" use xs2 scalar encoding "x0|zero".

### 2-arg .vv|.vx

**Instruction** | func2     | **func1** / Notes
:-------------: | :-------: | :-----------------------:
**Arithmetic**  | ...       | **000**
vadd            | 00 xxxxxx |
vsub            | 01 xxxxxx |
vrsub           | 02 xxxxxx |
veq             | 06 xxxxxx |
vne             | 07 xxxxxx |
vlt.{u}         | 08 xxxxxU |
vle.{u}         | 10 xxxxxU |
vgt.{u}         | 12 xxxxxU |
vge.{u}         | 14 xxxxxU |
vabsd.{u}       | 16 xxxxxU |
vmax.{u}        | 18 xxxxxU |
vmin.{u}        | 20 xxxxxU |
vadd3           | 24 xxxxxx |
**Arithmetic2** | ...       | **100**
vadds.{u}       | 00 xxxxxU |
vsubs.{u}       | 02 xxxxxU |
vaddw.{u}       | 04 xxxxxU |
vsubw.{u}       | 06 xxxxxU |
vacc.{u}        | 10 xxxxxU |
vpadd.{u}       | 12 xxxxxU | .v
vpsub.{u}       | 14 xxxxxU | .v
vhadd.{ur}      | 16 xxxxRU |
vhsub.{ur}      | 20 xxxxRU |
**Logical**     | ...       | **001**
vand            | 00 xxxxxx |
vor             | 01 xxxxxx |
vxor            | 02 xxxxxx |
vnot            | 03 xxxxxx | .v
vrev            | 04 xxxxxx |
vror            | 05 xxxxxx |
vclb            | 08 xxxxxx | .v
vclz            | 09 xxxxxx | .v
vcpop           | 10 xxxxxx | .v
vmv             | 12 xxxxxx | .v
vmvp            | 13 xxxxxx |
acset           | 16 xxxxxx |
actr            | 17 xxxxxx | .v
adwinit         | 18 xxxxxx |
**Shift**       | ...       | **010**
vsll            | 01 xxxxxx |
vsra            | 02 xxxxx0 |
vsrl            | 03 xxxxx1 |
vsha.{r}        | 08 xxxxR0 | +/- shamt
vshl.{r}        | 09 xxxxR1 | +/- shamt
vsrans{u}.{r}   | 16 xxxxRU | narrowing saturating (x2)
vsraqs{u}.{r}   | 24 xxxxRU | narrowing saturating (x4)
**Mul/Div**     | **...**   | **011**
vmul            | 00 xxxxxx |
vmuls           | 02 xxxxxU |
vmulw           | 04 xxxxxU |
vmulh.{ur}      | 08 xxxxRU |
vdmulh.{rn}     | 16 xxxxRN |
vmacc           | 20 xxxxxx |
vmadd           | 21 xxxxxx |
**Float**       | ...       | **101**
--reserved--    | xx xxxxxx |
**Shuffle**     | ...       | **110**
vslidevn        | 00 xxxxNN |
vslidehn        | 04 xxxxNN |
vslidevp        | 08 xxxxNN |
vslidehp        | 12 xxxxNN |
vsel            | 16 xxxxxx |
vevn            | 24 xxxxxx |
vodd            | 25 xxxxxx |
vevnodd         | 26 xxxxxx |
vzip            | 28 xxxxxx |
**Reserved7**   | ...       | **111**
--reserved--    | xx xxxxxx |

### 3-arg .vvv|.vxv

Instruction | func3 | Notes
:---------: | :---: | :-----------------------:
aconv       | 8     | scalar: sign
adwconv     | 10    | scalar: sign/type/swizzle

### Typeless

Operations that do not have a {.b,.h,.w} type have the same behavior regardless
of the size field (bitwise: vand, vnot, vor, vxor; move: vmv, vmvp). The tooling
convention is to use size=0b00 ".b" encoding.

### Vertical Modes

The ".tp" mode of vld or vst uses the four registers of ".m" in a vertical
structure, compared to other modes horizontal usage. The ".m" base update is a
single register width, vs 4x width for other modes. The usage model is four
"lines" being processed at the same time, vs a single line chained together in
other ".m" modes.

```
Horizontal
... AAAA BBBB CCCC DDDD ...

vs.

Vertical (".tp")
... AAAA ...
... BBBB ...
... CCCC ...
... DDDD ...
```

### Aliases

vneg.v ← vrsub.xv vd, vs1, zero \
vabs.v ← vabsd.vx vd, vs1, zero \
vwiden.v ← vaddw.vx vd, vs1, zero

## System Instructions

The execution model is designed towards OS-less and interrupt-less operation. A
machine will typically operate as run-to-completion of small restartable
workloads. A user/machine mode split is provided as a runtime convenience,
though there is no difference in access permissions between the modes.

31..28 | 27  | 26  | 25  | 24  | 23  | 22  | 21  | 20  | 19..15 | 14..12 | 11..7 | 6..2  | 1   | 0   | OP
:----: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :-: | :----: | :----: | :---: | :---: | :-: | :-: | :-:
0000   | PI  | PO  | PR  | PW  | SI  | SO  | SR  | SW  | 00000  | 000    | 00000 | 00011 | 1   | 1   | FENCE

<br>

31..28 | 27..24 | 23..20 | 19..15 | 14..12 | 11..7 | 6..2  | 1   | 0   | OP
:----: | :----: | :----: | :----: | :----: | :---: | :---: | :-: | :-: | :-----:
0000   | 0000   | 0000   | 00000  | 001    | 00000 | 00011 | 1   | 1   | FENCE.I

<br>

31..27 | 26..25 | 24..20 | 19..15 | 14..12 | 11..7 | 6..2  | 1   | 0   | OP
:----: | :----: | :----: | :----: | :----: | :---: | :---: | :-: | :-: | :-:
00100  | 11     | 00000  | xs1    | 000    | 00000 | 11101 | 1   | 1   | FLUSH
0001M  | sz     | xs2    | xs1    | 000    | xd    | 11101 | 1   | 1   | GET{MAX}VL
01111  | 00     | 00000  | xs1    | mode   | 00000 | 11101 | 1   | 1   | \[F,S,K,C\]LOG

<br>

31..20       | 19..15 | 14..12 | 11..7 | 6..2  | 1   | 0   | OP
:----------: | :----: | :----: | :---: | :---: | :-: | :-: | :----:
000000000001 | 00000  | 000    | 00000 | 11100 | 1   | 1   | EBREAK
001100000010 | 00000  | 000    | 00000 | 11100 | 1   | 1   | MRET
000010000000 | 00000  | 000    | 00000 | 11100 | 1   | 1   | MPAUSE
000001100000 | 00000  | 000    | 00000 | 11100 | 1   | 1   | ECTXSW
000001000000 | 00000  | 000    | 00000 | 11100 | 1   | 1   | EYIELD
000000100000 | 00000  | 000    | 00000 | 11100 | 1   | 1   | EEXIT
000000000000 | 00000  | 000    | 00000 | 11100 | 1   | 1   | ECALL

### Exit Cause

*   enum_IDLE = 0
*   enum_EBREAK = 1
*   enum_ECALL = 2
*   enum_EEXIT = 3
*   enum_EYIELD = 4
*   enum_ECTXSW = 5
*   enum_UNDEF_INST = (1u<<31) | 2
*   enum_USAGE_FAULT = (1u<<31) | 16

## Instruction Definitions

--------------------------------------------------------------------------------

### FLUSH

Cache clean and invalidate operations at the private level

**Encodings**

flushat xs1 \
flushall

**Operation**

```
Start = End = xs1
Line  = xs1
```
The instruction is a standard way of describing cache maintenance operations.

Type    | Visibility | System1           | System2
------- | ---------- | ----------------- | ---------------------
Private | Core       | Core L1           | Core L1 + Coherent L2

<br>

--------------------------------------------------------------------------------

### FENCE

Enforce memory ordering of loads and stores for external visibility.

**Encodings**

fence \[i|o|r|w\], \[i|o|r|w\] \
fence

**Operation**

```
PI predecessor I/O input
PO predecessor I/O output
PR predecessor memory read
PW predecessor memory write
<ordering between marked predecessors and successors>
SI successor I/O input
SO successor I/O output
SR successor memory read
SW successor memory write
```

Note: a simplified implementation may have the frontend stall until all
preceding operations are completed before permitting any trailing instruction to
be dispatched.

--------------------------------------------------------------------------------

### FENCE.I

Ensure subsequent instruction fetches observe prior data operations.

**Encodings**

fence.i

**Operation**

```
InvalidateInstructionCaches()
InvalidateInstructionPrefetchBuffers()
```

--------------------------------------------------------------------------------

### GETVL

Calculate the vector length.

**Encodings**

getvl.[b,h,w].x xd, xs1 \
getvl.[b,h,w].xx xd, xs1, xs2 \
getvl.[b,h,w].x.m xd, xs1 \
getvl.[b,h,w].xx.m xd, xs1, xs2

**Operation**

```
xd = min(vl.type.size, unsigned(xs1), xs2 ? unsigned(xs2) : ignore)
```

Find the minimum of the maximum vector length by type and the two input values.
If xs2 is zero (either x0 or register contents) then it is ignored (or
considered MaxInt), acting as a clamp less than maxvl.

Type | Instruction | Description
---- | ----------- | ----------------
00   | getvl.b     | 8bit lane count
01   | getvl.h     | 16bit lane count
10   | getvl.w     | 32bit lane count

--------------------------------------------------------------------------------

### GETMAXVL

Obtain the maximum vector length.

**Encodings**

getmaxvl.[b,h,w].{m} xd

**Operation**

```
xd = vl.type.size
```

Type | Instruction | Description
---- | ----------- | ----------------
00   | getmaxvl.b  | 8bit lane count
01   | getmaxvl.h  | 16bit lane count
10   | getmaxvl.w  | 32bit lane count

For a machine with 256bit SIMD registers:

*   getmaxvl.w = 8 lanes
*   getmaxvl.h = 16 lanes
*   getmaxvl.b = 32 lanes
*   getmaxvl.w.m = 32 lanes  &ensp; // multiply by 4 with strip mine.
*   getmaxvl.h.m = 64 lanes
*   getmaxvl.b.m = 128 lanes

--------------------------------------------------------------------------------

### ECALL

Execution call to supervisor OS.

**Encodings**

ecall

**Operation**

```
if (mode == User)
  mcause = enum_ECALL
  mepc = pc
  pc = mtvec
  mode = Machine
else
  mcause = enum_USAGE_FAULT
  mfault = pc
  EndExecution
```

--------------------------------------------------------------------------------

### EEXIT

Execution exit to supervisor OS.

**Encodings**

eexit

**Operation**

```
if (mode == User)
  mcause = enum_EEXIT
  mepc = pc
  pc = mtvec
  mode = Machine
else
  mcause = enum_USAGE_FAULT
  mfault = pc
  EndExecution
```

--------------------------------------------------------------------------------

### EYIELD

Synchronous execution switch to supervisor OS.

**Encodings**

eyield

**Operation**

```
if (mode == User)
  if (YIELD_REQUEST == 1)
    mcause = enum_EYIELD
    mepc = pc + 4  # advance to next instruction
    pc = mtvec
    mode = Machine
  else
    NOP  # pc = pc + 4
else
  mcause = enum_USAGE_FAULT
  mfault = pc
  EndExecution
```

YIELD_REQUEST refers to a signal the supervisor core sets to request a context
switch.

Note: use when MIE=0 eyield is inserted at synchronization points for
cooperative context switching.

--------------------------------------------------------------------------------

### ECTXSW

Asynchronous execution switch to supervisor OS.

**Encodings**

ectxsw

**Operation**

```
if (mode == User)
  mcause = enum_ECTXSW
  mepc = pc
  pc = mtvec
  mode = Machine
else
  mcause = enum_USAGE_FAULT
  mfault = pc
  EndExecution
```

--------------------------------------------------------------------------------

### EBREAK

Execution breakpoint to supervisor OS.

**Encodings**

ebreak

**Operation**

```
if (mode == User)
  mcause = enum_EBREAK
  mepc = pc
  pc = mtvec
  mode = Machine
else
  mcause = enum_UNDEF_INST
  mfault = pc
  EndExecution
```

--------------------------------------------------------------------------------

### MRET

Return from machine mode to user mode.

**Encodings**

mret

**Operation**

```
if (mode == Machine)
  pc = mepc
  mode = User
else
  mcause = enum_UNDEF_INST
  mepc = pc
  pc = mtvec
  mode = Machine
```

--------------------------------------------------------------------------------

### MPAUSE

Machine pause and release for next execution context.

**Encodings**

mpause

**Operation**

```
if (mode == Machine)
  EndExecution
else
  mcause = enum_UNDEF_INST
  mepc = pc
  pc = mtvec
  mode = Machine
```

--------------------------------------------------------------------------------

### VABSD

Absolute difference with unsigned result.

**Encodings**

vabsd.[b,h,w].{u}.vv.{m} vd, vs1, vs2 \
vabsd.[b,h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] > vs2[L] ? vs1[L] - vs2[L] : vs2[L] - vs1[L]
```

Note: for signed(INTx_MAX - INTx_MIN) the result will be UINTx_MAX.

--------------------------------------------------------------------------------

### VACC

Accumulates a value into a wider register.

**Encodings**

vacc.[h,w].{u}.vv.{m} vd, vs1, vs2 \
vacc.[h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  {vd+0}[L] = {vs1+0} + vs2.asHalfType[2*L+0]
  {vd+1}[L] = {vs1+1} + vs2.asHalfType[2*L+1]
```

--------------------------------------------------------------------------------

### VADD

Add operands.

**Encodings**

vadd.[b,h,w].vv.{m} vd, vs1, vs2 \
vadd.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] + vs2[L]
```

--------------------------------------------------------------------------------

### VADDS

Add operands with saturation.

**Encodings**

vadds.[b,h,w].{u}.vv.{m} vd, vs1, vs2 \
vadds.[b,h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = Saturate(vs1[L] + vs2[L])
```

--------------------------------------------------------------------------------

### VADDW

Add operands with widening.

**Encodings**

vaddw.[h,w].{u}.vv.{m} vd, vs1, vs2 \
vaddw.[h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  {vd+0}[L] = vs1.asHalfType[2*L+0] + vs2.asHalfType[2*L+0]
  {vd+1}[L] = vs1.asHalfType[2*L+1] + vs2.asHalfType[2*L+1]
```

--------------------------------------------------------------------------------

### VADD3

Add three operands.

**Encodings**

vadd3.[w].vv.{m} vd, vs1, vs2 \
vadd3.[w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in i32.typelen
  vd[L] = vd[L] + vs1[L] + vs2[L]
```

--------------------------------------------------------------------------------

### VAND

AND operands.

**Encodings**

vand.vv.{m} vd, vs1, vs2 \
vand.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] & vs2[L]
```

--------------------------------------------------------------------------------

### ACONV

Convolution ALU operation.

**Encodings**

aconv.vxv vd, vs1, xs2, vs3

Encoding 'aconv' uses a '1' in the unused 5th bit (b25) of vs2.

**Operation**

```
#  8b: 0123456789abcdef
# 32b: 048c 26ae 159d 37bf
assert(vd == 48)
N = is_simd512 ? 16 : is_simd256 ? 8 : assert(0)

func Interleave(Y,L):
  m = L % 4
  if (m == 0) (Y & ~3) + 0
  if (m == 1) (Y & ~3) + 2
  if (m == 2) (Y & ~3) + 1
  if (m == 3) (Y & ~3) + 3

# i32 += i8 x i8 (u*u, u*s, s*u, s*s)
for Y in [0..N-1]
  for X in [Start..Stop]
    for L in i8.typelen
      Data1 = {vs1+Y}.i8[4*X + L&3]  # 'transpose and broadcast'
      Data2 = {vs3+X-Start}.u8[L]
      {Accum+Interleave(Y,L)}[L / 4] +=
        ((signed(SData1,Data1{7:0}) + signed(Bias1{8:0})){9:0} *
         (signed(SData2,Data2{7:0}) + signed(Bias2{8:0})){9:0}){18:0}
```

Length (stop - start + 1) is in 32bit accumulator lane count, as all inputs will
horizontally reduce to this size.

The Start and Stop definition allows for a partial window of input values to be
transpose broadcast into the convolution unit.

Mode   | Mode | Usage
:----: | :--: | :-----------------------------------------------:
Common |      | Mode[1:0] Start[6:2] Stop[11:7]
s8     | 0    | SData2[31] SBias2[30:22] SData1[21] SBias1[20:12]

```
# SIMD256
acc.out = {v48..55}
narrow0 = {v0..7}
narrow1 = {v16..23}
narrow2 = {v32..39}
narrow3 = {v48..55}
wide0   = {v8..15}
wide1   = {v24..31}
wide2   = {v40..47}
wide3   = {v56..63}
```

### VCGET

Copy convolution accumulators into general registers.

**Encodings**

vcget vd

**Operation**

```
assert(vd == 48)
N = is_simd512 ? 15 : is_simd256 ? 7 : assert(0)
for Y in [0..N]
  vd{Y} = Accum{Y}
  Accum{Y} = 0

```

### ACSET

Copy general registers into convolution accumulators.

**Encodings**

acset.v vd, vs1

**Operation**

```
assert(vd == 48)
N = is_simd512 ? 15 : is_simd256 ? 7 : assert(0)
for Y in [0..N]
  Accum{Y} = vd{Y}
```

--------------------------------------------------------------------------------

### ACTR

Transpose a register group into the convolution accumulators.

**Encodings**

actr.[w].v.{m} vd, vs1

**Operation**

```
assert(vd in {v48})
assert(vs1 in {v0, v16, v32, v48}
for I in i32.typelen
  for J in i32.typelen
    ACCUM[J][I] = vs1[I][J]
```

--------------------------------------------------------------------------------

### VCLB

Count the leading bits.

**Encodings**

vclb.[b,h,w].v.{m} vd, vs1

**Operation**

```
MSB = 1 << (vtype.size - 1)
for L in Op.typelen
  vd[L] = vs1[L] & MSB ? CLZ(~vs1[L]) : CLZ(vs1[L])
```

Note: (clb - 1) is equivalent to __builtin_clrsb.

**clb examples**

```
clb.w(0xffffffff) = 32
clb.w(0xcfffffff) = 2
clb.w(0x80001000) = 1
clb.w(0x00007fff) = 17
clb.w(0x00000000) = 32
```

--------------------------------------------------------------------------------

### VCLZ

Count the leading zeros.

**Encodings**

vclz.[b,h,w].v.{m} vd, vs1

**Operation**

```
for L in Op.typelen
  vd[L] = CLZ(vs1[L])
```

Note: clz.[b,h,w](0) returns [8,16,32].

--------------------------------------------------------------------------------

### VDWCONV

Depthwise convolution 3-way multiply accumulate.

**Encodings**

vdwconv.vxv vd, vs1, x2, vs3 \
adwconv.vxv vd, vs1, x2, vs3

Encoding 'adwconv' uses a '1' in the unused 5th bit (b25) of vs2.

**Operation**

The vertical axis is typically tiled which requires preserving registers for
this functionality. The sparse formats require shuffles so that additional
registers of intermediate state are not required.

```
# quant8
{vs1+0,vs1+1,vs1+2} = Rebase({vs1}, Mode::RegBase)
{b0} = {vs3+0}.asByteType
{b1} = {vs3+1}.asByteType
{b2} = {vs3+2}.asByteType
if IsDenseFormat
  a0 = {vs1+0}.asByteType
  a1 = {vs1+1}.asByteType
  a2 = {vs1+2}.asByteType
if IsSparseFormat1  # [n-1,n,n+1]
  a0 = vslide_p({vs1+1}, {vs1+0}, 1).asByteType
  a1 = {vs1+0}.asByteType
  a2 = vslide_n({vs1+1}, {vs1+2}, 1).asByteType
if IsSparseFormat2  # [n,n+1,n+2]
  a0 = {vs1+0}.asByteType
  a1 = vslide_n({vs1+0}, {vs1+1}, 1).asByteType
  a2 = vslide_n({vs1+0}, {vs1+1}, 2).asByteType

#  8b: 0123456789abcdef
# 32b: 048c 26ae 159d 37bf
func Interleave(L):
  i = L % 4
  if (i == 0) 0
  if (i == 1) 2
  if (i == 2) 1
  if (i == 3) 3

for L in Op.typelen
  B = 4*L  # 8b --> 32b
  for i in [0..3]
    # int19_t multiply results
    # int23_t addition results
    # int32_t storage
    {dwacc+i}[L/4] +=
        (SData1(a0[B+i]) + bias1) * (SData2(b0[B+i]) + bias2) +
        (SData1(a1[B+i]) + bias1) * (SData2(b1[B+i]) + bias2) +
        (SData1(a2[B+i]) + bias1) * (SData2(b2[B+i]) + bias2)
  if is_vdwconv  // !adwconv
    for i in [0..3]
      {vd+i} = {dwacc+i}
```

Mode   | Encoding | Usage
:----: | :------: | :-----------------------------------------------:
Common | xs2      | Mode[1:0] Sparsity[3:2] RegBase[7:4]
q8     | 0        | SData2[31] SBias2[30:22] SData1[21] SBias1[20:12]

The Mode::Sparity sets the swizzling patterns.

Sparsity | Format  | Swizzle
:------: | :-----: | :---------:
b00      | Dense   | none
b01      | Sparse1 | [n-1,n,n+1]
b10      | Sparse2 | [n,n+1,n+2]

The Mode::RegBase allows for the start point of the 3 register group to allow
for cycling of [prev,curr,next] values.

RegBase | Prev    | Curr    | Next
:-----: | :-----: | :-----: | :-----:
b0000   | {vs1+0} | {vs1+1} | {vs1+2}
b0001   | {vs1+1} | {vs1+2} | {vs1+3}
b0010   | {vs1+2} | {vs1+3} | {vs1+4}
b0011   | {vs1+3} | {vs1+4} | {vs1+5}
b0100   | {vs1+4} | {vs1+5} | {vs1+6}
b0101   | {vs1+5} | {vs1+6} | {vs1+7}
b0110   | {vs1+6} | {vs1+7} | {vs1+8}
b0111   | {vs1+1} | {vs1+0} | {vs1+2}
b1000   | {vs1+1} | {vs1+2} | {vs1+0}
b1001   | {vs1+3} | {vs1+4} | {vs1+0}
b1010   | {vs1+5} | {vs1+6} | {vs1+0}
b1011   | {vs1+7} | {vs1+8} | {vs1+0}
b1100   | {vs1+2} | {vs1+0} | {vs1+1}
b1101   | {vs1+4} | {vs1+0} | {vs1+1}
b1110   | {vs1+6} | {vs1+0} | {vs1+1}
b1111   | {vs1+8} | {vs1+0} | {vs1+1}

Regbase supports upto 3x3 5x5 7x7 9x9, or use the extra horizontal range for
input latency hiding.

The vdwconv instruction includes a non-architectural state accumulator to
increase registerfile bandwidth. The dwinit instruction must be used to prepare
the depthwise accumulator for a sequence of dwconv instructions, and the
sequence must be dispatched without other instructions interleaved otherwise the
results will be unpredictable. Should other operations be required then a dwinit
must be inserted to resume the sequence.

In a context switch save where the accumulator must be saved alongside the
architectural simd registers, v0..63 are saved to thread stack or tcb and then a
vdwconv with vdup prepared zero inputs can be used to write the values to simd
registers and then saved to memory. In a context switch restore the values can
be loaded from memory and set in the accumulator registers using the dwinit
instruction.

### ADWINIT

Load the depthwise convolution accumulator state.

**Encodings**

adwinit.v vd, vs1

**Operation**

```
for L in Op.typelen
  {dwacc+0} = {vs1+0}[L]
  {dwacc+1} = {vs1+1}[L]
  {dwacc+2} = {vs1+2}[L]
  {dwacc+3} = {vs1+3}[L]
```

--------------------------------------------------------------------------------

### VDMULH

Saturating signed doubling multiply returning high half with optional rounding.

**Encodings**

vdmulh.[b,h,w].{r,rn}.vv.{m} vd, vs1, vs2 \
vdmulh.[b,h,w].{r,rn}.vx.{m} vd, vs1, xs2

**Operation**

```
SZ = vtype.size * 8
for L in Op.typelen
  LHS = SignExtend(vs1[L], 2*SZ)
  RHS = SignExtend(vs2[L], 2*SZ)
  MUL = LHS * RHS
  RND = R ? (N && MUL < 0 ? -(1<<(SZ-1)) : (1<<(SZ-1))) : 0
  vd[L] = SignedSaturation(2 * MUL + RND)[2*SZ-1:SZ]
```

Note: saturation is only needed for MaxNeg inputs (eg. 0x80000000).

Note: vdmulh.w.r.vx.m is used in ML activations so may be optimized by
implementations.

--------------------------------------------------------------------------------

### VDUP

Duplicate a scalar value into a vector register.

**Encodings**

vdup.[b,h,w].x.{m} vd, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = [xs2]
```

--------------------------------------------------------------------------------

### VEQ

Integer equal comparison.

**Encodings**

veq.[b,h,w].vv.{m} vd, vs1, vs2 \
veq.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] == vs2[L] ? 1 : 0
```

--------------------------------------------------------------------------------

### VEVN, VODD, VEVNODD

Even/odd of concatenated registers.

**Encodings**

vevn.[b,h,w].vv.{m} vd, vs1, vs2 \
vevn.[b,h,w].vx.{m} vd, vs1, xs2 \
vodd.[b,h,w].vv.{m} vd, vs1, vs2 \
vodd.[b,h,w].vx.{m} vd, vs1, xs2 \
vevnodd.[b,h,w].vv.{m} vd, vs1, vs2 \
vevnodd.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
M = Op.typelen / 2

if vevn || vevnodd
  {dst0} = {vd+0}
  {dst1} = {vd+1}
if vodd
  {dst1} = {vd+0}

if vevn || vevnodd
  for L in Op.typelen
    dst0[L] = L < M ? vs1[2 * L + 0] : vs2[2 * (L - M) + 0]  # even

if odd || vevnodd
  for L in Op.typelen
    dst1[L] = L < M ? vs1[2 * L + 1] : vs2[2 * (L - M) + 1]  # odd

where:
  vs1    = 0x33221100
  vs2    = 0x77665544
  {vd+0} = 0x66442200
  {vd+1} = 0x77553311
```

--------------------------------------------------------------------------------

#### VGE

Integer greater-than comparison.

**Encodings**

vgt.[b,h,w].{u}.vv.{m} vd, vs1, vs2 \
vgt.[b,h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] > vs2[L] ? 1 : 0
```

--------------------------------------------------------------------------------

#### VGT

Integer greater-than comparison.

**Encodings**

vgt.[b,h,w].{u}.vv.{m} vd, vs1, vs2 \
vgt.[b,h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] > vs2[L] ? 1 : 0
```

--------------------------------------------------------------------------------

### VHADD

Halving addition with optional rounding bit.

**Encodings**

vhadd.[b,h,w].{r,u,ur}.vv.{m} vd, vs1, vs2 \
vhadd.[b,h,w].{r,u,ur}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  if IsSigned()
    vd[L] = (signed(vs1[L]) + signed(vs2[L]) + R) >> 1
  else
    vd[L] = (unsigned(vs1[L]) + unsigned(vs2[L]) + R) >> 1
```

--------------------------------------------------------------------------------

### VHSUB

Halving subtraction with optional rounding bit.

**Encodings**

vhsub.[b,h,w].{r,u,ur}.vv.{m} vd, vs1, vs2 \
vhsub.[b,h,w].{r,u,ur}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  if IsSigned()
    vd[L] = (signed(vs1[L]) - signed(vs2[L]) + R) >> 1
  else
    vd[L] = (unsigned(vs1[L]) - unsigned(vs2[L]) + R) >> 1
```

--------------------------------------------------------------------------------

### VLD

Vector load from memory with optional post-increment by scalar.

**Encodings**

vld.[b,h,w].{p}.x.{m} vd, xs1 \
vld.[b,h,w].[l,p,s,lp,sp,tp].xx.{m} vd, xs1, xs2

**Operation**

```
addr = xs1
sm   = Op.m ? 4 : 1
len  = min(Op.typelen * sm, unsigned(xs2))
for M in Op.m
  for L in Op.typelen
    if !Op.bit.l || (L + M * Op.typelen) < len
      vd[L] = mem[addr + L].type
    else
      vd[L] = 0
  if (Op.bit.s)
    addr += xs2 * sizeof(type)
  else
    addr += Reg.bytes
if Op.bit.p
  if Op.bit.l && Op.bit.s                                  # .tp
    xs1 += Reg.bytes
  elif !Op.bit.l && !Op.bit.s && !{xs2}                    # .p.x
    xs1 += Reg.bytes * sm
  elif Op.bit.l                                            # .lp
    xs1 += len * sizeof(type)
  elif Op.bit.s                                            # .sp
    xs1 += xs2 * sizeof(type) * sm
  else                                                     # .p.xx
    xs1 += xs2 * sizeof(type)
```

--------------------------------------------------------------------------------

### VLE

Integer less-than-or-equal comparison.

**Encodings**

vle.[b,h,w].{u}.vv.{m} vd, vs1, vs2 \
vle.[b,h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] <= vs2[L] ? 1 : 0
```

--------------------------------------------------------------------------------

### VLT

Integer less-than comparison.

**Encodings**

vlt.[b,h,w].{u}.vv.{m} vd, vs1, vs2 \
vlt.[b,h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] < vs2[L] ? 1 : 0
```

--------------------------------------------------------------------------------

### VMACC

Multiply accumulate.

**Encodings**

vmacc.[b,h,w].vv.{m} vd, vs1, vs2 \
vmacc.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[N] += vs1[L] * vs2[L]
```

--------------------------------------------------------------------------------

### VMADD

Multiply add.

**Encodings**

vmadd.[b,h,w].vv.{m} vd, vs1, vs2 \
vmadd.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[N] = vd[L] * vs2[L] + vs1[L]
```

--------------------------------------------------------------------------------

### VMAX

Find the unsigned or signed maximum of two registers.

**Encodings**

vmax.[b,h,w].{u}.vv.{m} vd, vs1, vs2 \
vmax.[b,h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] > vs2[L] ? vs1[L] : vs2[L]
```

--------------------------------------------------------------------------------

### VMIN

Find the minimum of two registers.

**Encodings**

vmin.[b,h,w].{u}.vv.{m} vd, vs1, vs2 \
vmin.[b,h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] < vs2[L] ? vs1[L] : vs2[L]
```

--------------------------------------------------------------------------------

### VMUL

Multiply two registers.

**Encodings**

vmul.[b,h,w].vv.{m} vd, vs1, vs2 \
vmul.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] * vs2[L]
```

--------------------------------------------------------------------------------

### VMULS

Multiply with saturation two registers.

**Encodings**

vmuls.[b,h,w].{u}.vv.{m} vd, vs1, vs2 \
vmuls.[b,h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = Saturation(vs1[L] * vs2[L])
```

--------------------------------------------------------------------------------

### VMULW

Multiply with widening two registers.

**Encodings**

vmulw.[h,w].{u}.vv.{m} vd, vs1, vs2 \
vmulw.[h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  {vd+0}[L] = vs1.asHalfType[2*L+0] * vs2.asHalfType[2*L+0]
  {vd+1}[L] = vs1.asHalfType[2*L+1] * vs2.asHalfType[2*L+1]
```

--------------------------------------------------------------------------------

### VMULH

Multiply with widening two registers returning the high half.

**Encodings**

vmulh.[b,h,w].{u}.{r}.vv.{m} vd, vs1, vs2 \
vmulh.[b,h,w].{u}.{r}.vx.{m} vd, vs1, xs2

**Operation**

```
SZ = vtype.size * 8
RND = IsRounded ? 1<<(SZ-1) : 0
for L in Op.typelen
  if IsU()
    vd[L] = (unsigned(vs1[L]) * unsigned(vs2[L] + RND))[2*SZ-1:SZ]
  else if IsSU()
    vd[L] = (  signed(vs1[L]) * unsigned(vs2[L] + RND))[2*SZ-1:SZ]
  else
    vd[L] = (  signed(vs1[L]) *   signed(vs2[L] + RND))[2*SZ-1:SZ]
```

--------------------------------------------------------------------------------

### VMV

Move a register.

**Encodings**

vmv.v.{m} vd, vs1

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L]
```

Note: in the stripmined case an implemention may deliver more than one write per
cycle.

--------------------------------------------------------------------------------

### VMVP

Move a pair of registers.

**Encodings**

vmvp.vv.{m} vd, vs1, vs2 \
vmvp.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  {vd+0}[L] = vs1[L]
  {vd+1}[L] = vs2[L]
```

--------------------------------------------------------------------------------

### VNE

Integer not-equal comparison.

**Encodings**

vne.[b,h,w].vv.{m} vd, vs1, vs2 \
vne.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] != vs2[L] ? 1 : 0
```

--------------------------------------------------------------------------------

### VNOT

Bitwise NOT a register.

**Encodings**

vnot.v.{m} vd, vs1

**Operation**

```
for L in Op.typelen
  vd[L] = ~vs1[L]
```

--------------------------------------------------------------------------------

### VOR

OR two operands.

**Encodings**

vor.vv.{m} vd, vs1, vs2 \
vor.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] | vs2[L]
```

--------------------------------------------------------------------------------

### VPADD

Adds the lane pairs.

**Encodings**

vpadd.[h,w].{u}.v.{m} vd, vs1

**Operation**

```
if .v
  for L in Op.typelen
    vd[L] = (vs1.asHalfType[2 * L] + vs1.asHalfType[2 * L + 1])
```

--------------------------------------------------------------------------------

### VPSUB

Subtracts the lane pairs.

**Encodings**

vpsub.[h,w].{u}.v.{m} vd, vs1

**Operation**

```
if .v
  for L in Op.typelen
    vd[L] = (vs1.asHalfType[2 * L] - vs1.asHalfType[2 * L + 1])
```

--------------------------------------------------------------------------------

### VCPOP

Count the set bits.

**Encodings**

vcpop.[b,h,w].v.{m} vd, vs1

**Operation**

```
for L in Op.typelen
  vd[L] = CountPopulation(vs1[L])
```

--------------------------------------------------------------------------------

### VREV

Generalized reverse using bit ladder.

**Encodings**

vrev.[b,h,w].vv.{m} vd, vs1, vs2 \
vrev.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
N = vtype.bits - 1  # 7, 15, 31
shamt = xs2[4:0] & N
for L in Op.typelen
  r = vs1[L]
  if (shamt & 1)  r = ((r & 0x55..) << 1)  | ((r & 0xAA..) >> 1)
  if (shamt & 2)  r = ((r & 0x33..) << 2)  | ((r & 0xCC..) >> 2)
  if (shamt & 4)  r = ((r & 0x0F..) << 4)  | ((r & 0xF0..) >> 4)
  if (shamt & 8)  r = ((r & 0x00..) << 8)  | ((r & 0xFF..) >> 8)
  if (shamt & 16) r = ((r & 0x00..) << 16) | ((r & 0xFF..) >> 16)
  vd[L] = r
```

--------------------------------------------------------------------------------

### VROR

Logical rotate right.

**Encodings**

vror.[b,h,w].vv.{m} vd, vs1, vs2 \
vror.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
N = vtype.bits - 1  # 7, 15, 31
shamt = xs2[4:0] & N
for L in Op.typelen
  r = vs1[L]
  if (shamt & 1)  for (B in vtype.bits) r[B] = r[(N+1) % N]
  if (shamt & 2)  for (B in vtype.bits) r[B] = r[(N+2) % N]
  if (shamt & 4)  for (B in vtype.bits) r[B] = r[(N+4) % N]
  if (shamt & 8)  for (B in vtype.bits) r[B] = r[(N+8) % N]
  if (shamt & 16) for (B in vtype.bits) r[B] = r[(N+16) % N]
  vd[L] = r
```

--------------------------------------------------------------------------------

### VSHA, VSHL

Arithmetic and logical left/right shift with saturating shift amount and result.

**Encodings**

vsha.[b,h,w].{r}.vv.{m} vd, vs1, vs2

vshl.[b,h,w].{r}.vv.{m} vd, vs1, vs2

**Operation**

```
M = Op.size  # 8, 16, 32
N = [8->3, 16->4, 32->5][Op.size]
SHSAT[L] = vs2[L][M-1:N] != 0
SHAMT[L] = vs2[L][N-1:0]
RND  = R && SHAMT ? 1 << (SHAMT-1) : 0
RND -= N && (vs1[L] < 0) ? 1 : 0
SZ = sizeof(src.type) * 8 * (W ? 2 : 1)
RESULT_NEG = (vs1[L] <<[<] SHAMT[L])[SZ-1:0]  // !A "<<<" logical shift
RESULT_NEG = S ? Saturate(RESULT_POS, SHSAT[L]) : RESULT_NEG
RESULT_POS = ((vs1[L] + RND) >>[>] SHAMT[L])  // !A ">>>" logical shift
RESULT_POS = S ? Saturate(RESULT_NEG, SHSAT[L]) : RESULT_POS
xd[L] = SHAMT[L] >= 0 ? RESULT_POS : RESULT_NEG
```

--------------------------------------------------------------------------------

### VSEL

Select lanes from two operands with vector selection boolean.

**Encodings**

vsel.[b,h,w].vv.{m} vd, vs1, vs2 \
vsel.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L].bit(0) ? vd[L] : vs2[L]
```

--------------------------------------------------------------------------------

### VSLL

Logical left shift.

**Encodings**

vsll.[b,h,w].vv.{m} vd, vs1, vs2 \
vsll.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
N = [8->3, 16->4, 32->5][Op.size]
xd[L] = vs1[L] <<< vs2[L][N-1:0]
```

--------------------------------------------------------------------------------

### VSLIDEN

Slide next register by index.

**Encodings**

vslidehn.[b,h,w].[1,2,3,4].vv.m vd, vs1, vs2 \
vslidevn.[b,h,w].[1,2,3,4].vv.m vd, vs1, vs2 \
vslidehn.[b,h,w].[1,2,3,4].vx.m vd, vs1, xs2 \
vslidevn.[b,h,w].[1,2,3,4].vx.m vd, vs1, xs2

**Operation**

```
assert vd != vs1 && vd != vs2
if Op.h
  va = {{vs1+3},{vs1+2},{vs1+1},{vs1+0}}
  vb = {{vs2+0},{vs1+3},{vs1+2},{vs1+1}}
if Op.v
  va = {{vs1+3},{vs1+2},{vs1+1},{vs1+0}}
  vb = {{vs2+3},{vs2+2},{vs2+1},{vs2+0}}
for M in Op.m
  for L in Op.typelen
    if (L + index < Op.typelen)
      vd[L] = va[M][L + index]
    else
      vd[L] = vb[M][L + index - Op.typelen]
```

--------------------------------------------------------------------------------

### VSLIDEP

Slide previous register by index.

**Encodings**

vslidehp.[b,h,w].[1,2,3,4].vv.m vd, vs1, vs2 \
vslidevp.[b,h,w].[1,2,3,4].vv.m vd, vs1, vs2

**Operation**

```
assert vd != vs1 && vd != vs2
if Op.h
  va = {{vs1+3},{vs1+2},{vs1+1},{vs1+0}}
  vb = {{vs2+0},{vs1+3},{vs1+2},{vs1+1}}
if Op.v
  va = {{vs1+3},{vs1+2},{vs1+1},{vs1+0}}
  vb = {{vs1+2},{vs1+1},{vs1+0},{vs2+3}}
for M in Op.m
  for L in Op.typelen
    if (L >= index)
      vd[L] = va[M][L - index]
    else
      vd[L] = vb[M][Op.typelen + L - index]
```

--------------------------------------------------------------------------------

### VSRA, VSRL

Arithmetic and logical right shift.

**Encodings**

vsra.[b,h,w].vv.{m} vd, vs1, vs2 \
vsra.[b,h,w].vx.{m} vd, vs1, xs2

vsrl.[b,h,w].vv.{m} vd, vs1, vs2 \
vsrl.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
N = Op.size[8->3, 16->4, 32->5]
xd[L] = vs1[L] >>[>] vs2[L][N-1:0]
```

--------------------------------------------------------------------------------

### VSRANS, VSRANSU

Arithmetic right shift with rounding and signed/unsigned saturation.

**Encodings**

vsrans{u}.[b,h,w].{r}.vv.{m} vd, vs1, vs2 \
vsrans{u}.[b,h,w].{r}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  N = [8->3, 16->4, 32->5][Op.size]
  SHAMT[L] = vs2[L][2*N-1:0]  # source size index
  RND  = R && SHAMT ? 1 << (SHAMT-1) : 0
  RND -= N && (vs1[L] < 0) ? 1 : 0
  vd[L+0] = Saturate({vs1+0}[L/2] + RND, u) >>[>] SHAMT
  vd[L+1] = Saturate({vs1+1}[L/2] + RND, u) >>[>] SHAMT
```

Note: vsrans.[b,h].vx.m are used in ML activations so may be optimized by
implementations.

--------------------------------------------------------------------------------

### VSRAQS

Arithmetic quarter narrowing right shift with rounding and signed/unsigned
saturation.

**Encodings**

vsraqs{u}.[b,h].{r}.vv.{m} vd, vs1, vs2 \
vsraqs{u}.[b,h].{r}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in i32.typelen
  SHAMT[L] = vs2[L][4:0]
  RND  = R && SHAMT ? 1 << (SHAMT-1) : 0
  RND -= N && (vs1[L] < 0) ? 1 : 0
  vd[L+0] = Saturate({vs1+0}[L/4] + RND, u) >>[>] SHAMT
  vd[L+1] = Saturate({vs1+2}[L/4] + RND, u) >>[>] SHAMT
  vd[L+2] = Saturate({vs1+1}[L/4] + RND, u) >>[>] SHAMT
  vd[L+3] = Saturate({vs1+3}[L/4] + RND, u) >>[>] SHAMT
```

Note: The register interleaving is [0,2,1,3] and not [0,1,2,3] as this matches
vconv/vdwconv requirements, and one vsrxqs is the same as two chained vsrxns.

--------------------------------------------------------------------------------

### VRSUB

Reverse subtract two operands.

**Encodings**

vrsub.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = xs2[L] - vs1[L]
```

--------------------------------------------------------------------------------

### VSUB

Subtract two operands.

**Encodings**

vsub.[b,h,w].vv.{m} vd, vs1, vs2 \
vsub.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] - vs2[L]
```

--------------------------------------------------------------------------------

### VSUBS

Subtract two operands with saturation.

**Encodings**

vsubs.[b,h,w].{u}.vv.{m} vd, vs1, vs2 \
vsubs.[b,h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = Saturate(vs1[L] - vs2[L])
```

--------------------------------------------------------------------------------

### VSUBW

Subtract two operands with widening.

**Encodings**

vsubw.[h,w].{u}.vv.{m} vd, vs1, vs2 \
vsubw.[h,w].{u}.vx.{m} vd, vs1, xs2

**Operation**`

```
for L in Op.typelen
  {vd+0}[L] = vs1.asHalfType[2*L+0] - vs2.asHalfType[2*L+0]
  {vd+1}[L] = vs1.asHalfType[2*L+1] - vs2.asHalfType[2*L+1]
```

--------------------------------------------------------------------------------

### VST

Vector store to memory with optional post-increment by scalar.

**Encodings**

vst.[b,h,w].{p}.x.{m} vd, xs1 \
vst.[b,h,w].[l,p,s,lp,sp,tp].xx.{m} vd, xs1, xs2

**Operation**

```
addr = xs1
sm   = Op.m ? 4 : 1
len  = min(Op.typelen * sm, unsigned(xs2))
for M in Op.m
  for L in Op.typelen
    if !Op.bit.l || (L + M * Op.typelen) < len
      mem[addr + L].type = vd[L]
  if (Op.bit.s)
    addr += xs2 * sizeof(type)
  else
    addr += Reg.bytes
if Op.bit.p
  if Op.bit.l && Op.bit.s                                  # .tp
    xs1 += Reg.bytes
  elif !Op.bit.l && !Op.bit.s && !{xs2}                    # .p.x
    xs1 += Reg.bytes * sm
  elif Op.bit.l                                            # .lp
    xs1 += len * sizeof(type)
  elif Op.bit.s                                            # .sp
    xs1 += xs2 * sizeof(type) * sm
  else                                                     # .p.xx
    xs1 += xs2 * sizeof(type)
```

--------------------------------------------------------------------------------

### VSTQ

Vector store quads to memory with optional post-increment by scalar.

**Encodings**

vstq.[b,h,w].[s,sp].xx.{m} vd, xs1, xs2

**Operation**

```
addr = xs1
sm   = Op.m ? 4 : 1
for M in Op.m
  for Q in 0 to 3
    for L in Op.typelen / 4
      mem[addr + L].type = vd[L + Q * Op.typelen / 4]
      addr += xs2 * sizeof(type)
if Op.bit.p
  xs1 += xs2 * sizeof(type) * sm
```

Note: This is principally for storing the results of vconv after 32b to 8b
reduction.

--------------------------------------------------------------------------------

### VXOR

XOR two operands.

**Encodings**

vxor.vv.{m} vd, vs1, vs2 \
vxor.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
for L in Op.typelen
  vd[L] = vs1[L] ^ vs2[L]
```

--------------------------------------------------------------------------------

### VZIP

Interleave even/odd lanes of two operands.

**Encodings**

vzip.[b,h,w].vv.{m} vd, vs1, vs2 \
vzip.[b,h,w].vx.{m} vd, vs1, xs2

**Operation**

```
index = Is(a=>0, b=>1)
for L in Op.typelen
  M = L / 2
  N = L / 2 + Op.typelen / 2
  {vd+0}[L] = L & 1 ? vs2[M] : vs1[M]
  {vd+1}[L] = L & 1 ? vs2[N] : vs1[N]

where:
  vs1    = 0x66442200
  vs2    = 0x77553311
  {vd+0} = 0x33221100
  {vd+1} = 0x77665544
```

Note: vd must not be in the range of vs1 or vs2.

--------------------------------------------------------------------------------

### FLOG, SLOG, CLOG, KLOG

Log a register in a printf contract.

**Encodings**

flog rs1 &ensp; // mode=0, “printf” formatted command, rs1=(context) \
slog rs1 &ensp; // mode=1, scalar log \
clog rs1 &ensp; // mode=2, character log \
klog rs1 &ensp; // mode=3, const string log

**Operation**

A number of arguments are sent with SLOG or CLOG, and then a FLOG operation
closes the packet and may emit a timestamp and context data like ASID. A
receiving tool can construct messages, e.g. XML records per printf stream, by
collecting the arguments as they arrive in a variable length buffer, and closing
the record when the FLOG instruction arrives.

A transport layer may choose to encode in the flog format footer the preceding
count of arguments or bytes sent. This is so that detection of payload errors or
hot connections are possible.

The SLOG instruction will send a payload packet represented by the starting
memory location.

The CLOG instruction will send a multiple 32-bit packet message of a character
stream. The packet message will close when a zero character is detected. A
single character may be sent in a 32bit packet.

**Pseudo code**

```
const uint8_t p[] = "text message";
printf(“Test %s\n”, p);
    KLOG p
    FLOG &fmt
```

```
printf(“Test”);
    FLOG &fmt
```

```
print(“Test %d\n”, result_int);
    SLOG result_int
    FLOG &fmt
```

```
printf(“Test %d %f %s %s %s\n”, 123, "abc", "1234", “789AB”);
    SLOG 123
    CLOG ‘abc\0’
    CLOG ‘1234’ CLOG ‘\0’
    CLOG ‘789A’ CLOG ‘B\0’
    FLOG &fmt
```
