/*
 * Copyright 2023 Google LLC
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cstdio>

#include "tests/kelvin_isa/kelvin_test.h"

int main() {
  test_alu_w_vv("vand.vv", 0x55ffcc5a, 0xaa015566, 0x00014442);

  test_alu_b_v("vclb.b.v", 0x00, 8);
  test_alu_b_v("vclb.b.v", 0x80, 1);
  test_alu_b_v("vclb.b.v", 0x10, 3);
  test_alu_b_v("vclb.b.v", 0x90, 1);
  test_alu_b_v("vclb.b.v", 0x55, 1);
  test_alu_b_v("vclb.b.v", 0xd5, 2);
  test_alu_b_v("vclb.b.v", 0xaa, 1);
  test_alu_b_v("vclb.b.v", 0x2a, 2);
  test_alu_b_v("vclb.b.v", 0xf0, 4);
  test_alu_b_v("vclb.b.v", 0x70, 1);
  test_alu_b_v("vclb.b.v", 0xff, 8);
  test_alu_b_v("vclb.b.v", 0x7f, 1);
  test_alu_h_v("vclb.h.v", 0x0000, 16);
  test_alu_h_v("vclb.h.v", 0x8000, 1);
  test_alu_h_v("vclb.h.v", 0x0010, 11);
  test_alu_h_v("vclb.h.v", 0x8010, 1);
  test_alu_h_v("vclb.h.v", 0x0800, 4);
  test_alu_h_v("vclb.h.v", 0x8800, 1);
  test_alu_h_v("vclb.h.v", 0x5555, 1);
  test_alu_h_v("vclb.h.v", 0xd555, 2);
  test_alu_h_v("vclb.h.v", 0xaaaa, 1);
  test_alu_h_v("vclb.h.v", 0x2aaa, 2);
  test_alu_h_v("vclb.h.v", 0xff00, 8);
  test_alu_h_v("vclb.h.v", 0x7f00, 1);
  test_alu_h_v("vclb.h.v", 0xffff, 16);
  test_alu_h_v("vclb.h.v", 0x7fff, 1);
  test_alu_w_v("vclb.w.v", 0x00000000, 32);
  test_alu_w_v("vclb.w.v", 0x80000000, 1);
  test_alu_w_v("vclb.w.v", 0x00000010, 27);
  test_alu_w_v("vclb.w.v", 0x80000010, 1);
  test_alu_w_v("vclb.w.v", 0x00000800, 20);
  test_alu_w_v("vclb.w.v", 0x80000800, 1);
  test_alu_w_v("vclb.w.v", 0x00002000, 18);
  test_alu_w_v("vclb.w.v", 0x80002000, 1);
  test_alu_w_v("vclb.w.v", 0x01000000, 7);
  test_alu_w_v("vclb.w.v", 0x81000000, 1);
  test_alu_w_v("vclb.w.v", 0x55555555, 1);
  test_alu_w_v("vclb.w.v", 0xd5555555, 2);
  test_alu_w_v("vclb.w.v", 0xaaaaaaaa, 1);
  test_alu_w_v("vclb.w.v", 0x2aaaaaaa, 2);
  test_alu_w_v("vclb.w.v", 0xffff0000, 16);
  test_alu_w_v("vclb.w.v", 0x7fff0000, 1);
  test_alu_w_v("vclb.w.v", 0xffffffff, 32);
  test_alu_w_v("vclb.w.v", 0x7fffffff, 1);

  test_alu_b_v("vclz.b.v", 0x00, 8);
  test_alu_b_v("vclz.b.v", 0x10, 3);
  test_alu_b_v("vclz.b.v", 0x55, 1);
  test_alu_b_v("vclz.b.v", 0xaa, 0);
  test_alu_h_v("vclz.h.v", 0x0000, 16);
  test_alu_h_v("vclz.h.v", 0x0010, 11);
  test_alu_h_v("vclz.h.v", 0x0800, 4);
  test_alu_h_v("vclz.h.v", 0x5555, 1);
  test_alu_h_v("vclz.h.v", 0xaaaa, 0);
  test_alu_w_v("vclz.w.v", 0x00000000, 32);
  test_alu_w_v("vclz.w.v", 0x00000010, 27);
  test_alu_w_v("vclz.w.v", 0x00000800, 20);
  test_alu_w_v("vclz.w.v", 0x00002000, 18);
  test_alu_w_v("vclz.w.v", 0x01000000, 7);
  test_alu_w_v("vclz.w.v", 0x55555555, 1);
  test_alu_w_v("vclz.w.v", 0xaaaaaaaa, 0);

  test_alu_w_v("vnot.v", 0x55ffcc5a, 0xaa0033a5);
  test_alu_w_v("vnot.v", 0xa05f1248, 0x5fa0edb7);

  test_alu_w_vv("vor.vv", 0x55555555, 0xaaaaaaaa, 0xffffffff);
  test_alu_w_vv("vor.vv", 0xa05f1248, 0x5fa02481, 0xffff36c9);
  test_alu_w_vv("vor.vv", 0x12345678, 0x87654321, 0x97755779);

  test_alu_b_v("vcpop.b.v", 0x00, 0);
  test_alu_b_v("vcpop.b.v", 0x10, 1);
  test_alu_b_v("vcpop.b.v", 0x12, 2);
  test_alu_b_v("vcpop.b.v", 0x55, 4);
  test_alu_b_v("vcpop.b.v", 0xaa, 4);
  test_alu_b_v("vcpop.b.v", 0x7e, 6);
  test_alu_b_v("vcpop.b.v", 0x7f, 7);
  test_alu_b_v("vcpop.b.v", 0xfe, 7);
  test_alu_b_v("vcpop.b.v", 0xff, 8);
  test_alu_h_v("vcpop.h.v", 0x0000, 0);
  test_alu_h_v("vcpop.h.v", 0x0010, 1);
  test_alu_h_v("vcpop.h.v", 0x0800, 1);
  test_alu_h_v("vcpop.h.v", 0x1234, 5);
  test_alu_h_v("vcpop.h.v", 0x5555, 8);
  test_alu_h_v("vcpop.h.v", 0xaaaa, 8);
  test_alu_h_v("vcpop.h.v", 0x7ffe, 14);
  test_alu_h_v("vcpop.h.v", 0x7fff, 15);
  test_alu_h_v("vcpop.h.v", 0xfffe, 15);
  test_alu_h_v("vcpop.h.v", 0xffff, 16);
  test_alu_w_v("vcpop.w.v", 0x00000000, 0);
  test_alu_w_v("vcpop.w.v", 0x00000010, 1);
  test_alu_w_v("vcpop.w.v", 0x00000800, 1);
  test_alu_w_v("vcpop.w.v", 0x00702000, 4);
  test_alu_w_v("vcpop.w.v", 0x0100060a, 5);
  test_alu_w_v("vcpop.w.v", 0x12345678, 13);
  test_alu_w_v("vcpop.w.v", 0x55555555, 16);
  test_alu_w_v("vcpop.w.v", 0xaaaaaaaa, 16);
  test_alu_w_v("vcpop.w.v", 0x7ffffffe, 30);
  test_alu_w_v("vcpop.w.v", 0x7fffffff, 31);
  test_alu_w_v("vcpop.w.v", 0xfffffffe, 31);
  test_alu_w_v("vcpop.w.v", 0xffffffff, 32);

  test_alu_w_vx("vrev.b.vx", 0x12345678, 0x00, 0x12345678);
  test_alu_w_vx("vrev.b.vx", 0x12345678, 0x01, 0x2138a9b4);
  test_alu_w_vx("vrev.b.vx", 0x12345678, 0x02, 0x48c159d2);
  test_alu_w_vx("vrev.b.vx", 0x12345678, 0x04, 0x21436587);
  test_alu_w_vx("vrev.b.vx", 0x12345678, 0x08, 0x12345678);
  test_alu_w_vx("vrev.b.vx", 0x12345678, 0x10, 0x12345678);
  test_alu_w_vx("vrev.b.vx", 0x12345678, 0x1f, 0x482c6a1e);
  test_alu_w_vx("vrev.b.vx", 0x12345678, 0xe0, 0x12345678);

  test_alu_w_vx("vrev.h.vx", 0x12345678, 0x00, 0x12345678);
  test_alu_w_vx("vrev.h.vx", 0x12345678, 0x01, 0x2138a9b4);
  test_alu_w_vx("vrev.h.vx", 0x12345678, 0x02, 0x48c159d2);
  test_alu_w_vx("vrev.h.vx", 0x12345678, 0x04, 0x21436587);
  test_alu_w_vx("vrev.h.vx", 0x12345678, 0x08, 0x34127856);
  test_alu_w_vx("vrev.h.vx", 0x12345678, 0x10, 0x12345678);
  test_alu_w_vx("vrev.h.vx", 0x12345678, 0x07, 0x482c6a1e);
  test_alu_w_vx("vrev.h.vx", 0x12345678, 0x0f, 0x2c481e6a);
  test_alu_w_vx("vrev.h.vx", 0x12345678, 0x1f, 0x2c481e6a);
  test_alu_w_vx("vrev.h.vx", 0x12345678, 0xe0, 0x12345678);

  test_alu_w_vx("vrev.w.vx", 0x12345678, 0x00, 0x12345678);
  test_alu_w_vx("vrev.w.vx", 0x12345678, 0x01, 0x2138a9b4);
  test_alu_w_vx("vrev.w.vx", 0x12345678, 0x02, 0x48c159d2);
  test_alu_w_vx("vrev.w.vx", 0x12345678, 0x04, 0x21436587);
  test_alu_w_vx("vrev.w.vx", 0x12345678, 0x08, 0x34127856);
  test_alu_w_vx("vrev.w.vx", 0x12345678, 0x10, 0x56781234);
  test_alu_w_vx("vrev.w.vx", 0x12345678, 0x07, 0x482c6a1e);
  test_alu_w_vx("vrev.w.vx", 0x12345678, 0x0f, 0x2c481e6a);
  test_alu_w_vx("vrev.w.vx", 0x12345678, 0x1f, 0x1e6a2c48);
  test_alu_w_vx("vrev.w.vx", 0x12345678, 0xe0, 0x12345678);

  test_alu_w_vx("vror.b.vx", 0x23456789, 0x00, 0x23456789);
  test_alu_w_vx("vror.b.vx", 0x23456789, 0x01, 0x91a2b3c4);
  test_alu_w_vx("vror.b.vx", 0x23456789, 0x02, 0xc851d962);
  test_alu_w_vx("vror.b.vx", 0x23456789, 0x04, 0x32547698);
  test_alu_w_vx("vror.b.vx", 0x23456789, 0x08, 0x23456789);
  test_alu_w_vx("vror.b.vx", 0x23456789, 0x10, 0x23456789);
  test_alu_w_vx("vror.b.vx", 0x23456789, 0x1f, 0x468ace13);
  test_alu_w_vx("vror.b.vx", 0x23456789, 0xe0, 0x23456789);

  test_alu_w_vx("vror.h.vx", 0x23456789, 0x00, 0x23456789);
  test_alu_w_vx("vror.h.vx", 0x23456789, 0x01, 0x91a2b3c4);
  test_alu_w_vx("vror.h.vx", 0x23456789, 0x02, 0x48d159e2);
  test_alu_w_vx("vror.h.vx", 0x23456789, 0x04, 0x52349678);
  test_alu_w_vx("vror.h.vx", 0x23456789, 0x08, 0x45238967);
  test_alu_w_vx("vror.h.vx", 0x23456789, 0x10, 0x23456789);
  test_alu_w_vx("vror.h.vx", 0x23456789, 0x07, 0x8a4612cf);
  test_alu_w_vx("vror.h.vx", 0x23456789, 0x0f, 0x468acf12);
  test_alu_w_vx("vror.h.vx", 0x23456789, 0x1f, 0x468acf12);
  test_alu_w_vx("vror.h.vx", 0x23456789, 0xe0, 0x23456789);

  test_alu_w_vx("vror.w.vx", 0x23456789, 0x00, 0x23456789);
  test_alu_w_vx("vror.w.vx", 0x23456789, 0x01, 0x91a2b3c4);
  test_alu_w_vx("vror.w.vx", 0x23456789, 0x02, 0x48d159e2);
  test_alu_w_vx("vror.w.vx", 0x23456789, 0x04, 0x92345678);
  test_alu_w_vx("vror.w.vx", 0x23456789, 0x08, 0x89234567);
  test_alu_w_vx("vror.w.vx", 0x23456789, 0x10, 0x67892345);
  test_alu_w_vx("vror.w.vx", 0x23456789, 0x07, 0x12468acf);
  test_alu_w_vx("vror.w.vx", 0x23456789, 0x0f, 0xcf12468a);
  test_alu_w_vx("vror.w.vx", 0x23456789, 0x1f, 0x468acf12);
  test_alu_w_vx("vror.w.vx", 0x23456789, 0xe0, 0x23456789);

  test_alu_w_vv("vxor.vv", 0x55555555, 0xaaaaaaaa, 0xffffffff);
  test_alu_w_vv("vxor.vv", 0x55555555, 0x5555ffff, 0x0000aaaa);
  test_alu_w_vv("vxor.vv", 0xa05f1248, 0x5fa02481, 0xffff36c9);
  test_alu_w_vv("vxor.vv", 0x12345678, 0x87654321, 0x95511559);

  return 0;
}
