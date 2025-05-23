/***************************************************************************************************
 * Copyright (c) 2023 - 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: BSD-3-Clause
 *
 * Redistribution and use in source and binary forms, with or without
 * modification, are permitted provided that the following conditions are met:
 *
 * 1. Redistributions of source code must retain the above copyright notice, this
 * list of conditions and the following disclaimer.
 *
 * 2. Redistributions in binary form must reproduce the above copyright notice,
 * this list of conditions and the following disclaimer in the documentation
 * and/or other materials provided with the distribution.
 *
 * 3. Neither the name of the copyright holder nor the names of its
 * contributors may be used to endorse or promote products derived from
 * this software without specific prior written permission.
 *
 * THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
 * AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
 * IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
 * DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
 * FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
 * DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
 * SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
 * CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
 * OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
 * OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.
 *
 **************************************************************************************************/
/** Common algorithms on (hierarchical) tensors */

#pragma once

#include <cute/config.hpp>
#include <cute/tensor_impl.hpp>

namespace cute
{

//
// for_each
//

template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
for_each(Tensor<Engine,Layout> const& tensor, UnaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor); ++i) {
    op(tensor(i));
  }
}

template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
for_each(Tensor<Engine,Layout>& tensor, UnaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor); ++i) {
    op(tensor(i));
  }
}

// Accept mutable temporaries
template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
for_each(Tensor<Engine,Layout>&& tensor, UnaryOp&& op)
{
  return for_each(tensor, op);
}

//
// transform
//

// Similar to std::transform but does not return number of elements affected
template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<Engine,Layout>& tensor, UnaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor); ++i) {
    tensor(i) = op(tensor(i));
  }
}

// Accept mutable temporaries
template <class Engine, class Layout, class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<Engine,Layout>&& tensor, UnaryOp&& op)
{
  return transform(tensor, op);
}

// Similar to std::transform transforms one tensors and assigns it to another
template <class EngineIn, class LayoutIn,
          class EngineOut, class LayoutOut,
          class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<EngineIn, LayoutIn > const& tensor_in,
          Tensor<EngineOut,LayoutOut>      & tensor_out,
          UnaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor_in); ++i) {
    tensor_out(i) = op(tensor_in(i));
  }
}

// Accept mutable temporaries
template <class EngineIn, class LayoutIn,
          class EngineOut, class LayoutOut,
          class UnaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<EngineIn, LayoutIn > const& tensor_in,
          Tensor<EngineOut,LayoutOut>     && tensor_out,
          UnaryOp&& op)
{
  return transform(tensor_in, tensor_out, op);
}

// Similar to std::transform with a binary operation
// Takes two tensors as input and one tensor as output.
// Applies the binary_op to tensor_in1 and tensor_in2 and
// assigns it to tensor_out
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut,
          class BinaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>      & tensor_out,
          BinaryOp&& op)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor_in1); ++i) {
    tensor_out(i) = op(tensor_in1(i), tensor_in2(i));
  }
}

// Accept mutable temporaries
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut,
          class BinaryOp>
CUTE_HOST_DEVICE constexpr
void
transform(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>     && tensor_out,
          BinaryOp&& op)
{
  return transform(tensor_in1, tensor_in2, tensor_out, op);
}

template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut>
CUTE_DEVICE constexpr
void
transform2(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>      & tensor_out)
{
  // Batch logic operations
  int B = 4;
  int SZ = size(tensor_in1);

  Tensor t1 = recast<uint32_t>(tensor_in1);
  Tensor t2 = recast<uint32_t>(tensor_in2);
  Tensor t3 = recast<uint32_t>(tensor_out);

  CUTE_UNROLL
  for (int i = 0; i < SZ / B; i++) {
    uint32_t a = t1(i);
    uint32_t b = t2(i);
    uint32_t s = a & 0x80808080;
    uint32_t sub = (b & 0x80808080) >> 7;
    t1(i) = (((a - sub) >> 1) & 0x3f3f3f3f) | s;
    uint32_t c = __byte_perm(t1(i), t2(i), 0x1504);
    uint32_t d = __byte_perm(t1(i), t2(i), 0x3726);
    t3((B/2)*i) = c;
    t3((B/2)*i+1) = d;
  }

  /*
  // Batch logic operations
  int B = 4;
  int SZ = size(tensor_in1);

  Tensor t1 = recast<uint32_t>(tensor_in1);
  Tensor t2 = recast<uint32_t>(tensor_in2);
  Tensor t3 = recast<uint32_t>(tensor_out);

  CUTE_UNROLL
  for (int i = 0; i < SZ / B; i++) {
    uint32_t a = t1(i);
    uint32_t b = t2(i);
    uint32_t s = a & 0x80808080;
    uint32_t sub = (b & 0x80808080) >> 7;
    t1(i) = (((a - sub) >> 1) & 0x3f3f3f3f) | s;
    uint32_t c = __byte_perm(t1(i), t2(i), 0x1504);
    uint32_t d = __byte_perm(t1(i), t2(i), 0x3726);
    t3((B/2)*i) = c;
    t3((B/2)*i+1) = d;
  }
  */

  /*
  // Batch logic operations
  int B = 4;
  int SZ = size(tensor_in1);

  Tensor t1 = recast<uint32_t>(tensor_in1);
  Tensor t2 = recast<uint32_t>(tensor_in2);
  Tensor t1_ = recast<uint8_t>(tensor_in1);
  Tensor t2_ = recast<uint8_t>(tensor_in2);

  CUTE_UNROLL
  for (int i = 0; i < SZ / B; i++) {
    uint32_t a = t1(i);
    uint32_t b = t2(i);
    uint32_t s = a & 0x80808080;
    uint32_t sub = ((b >> 7) ^ a) & 0x01010101;
    t1(i) = (((a - sub) >> 1) & 0x3f3f3f3f) | s;

    tensor_out(B*i+0) = cutlass::half_t::bitcast((t1_(B*i+0) << 8) | t2_(B*i+0));
    tensor_out(B*i+1) = cutlass::half_t::bitcast((t1_(B*i+1) << 8) | t2_(B*i+1));
    tensor_out(B*i+2) = cutlass::half_t::bitcast((t1_(B*i+2) << 8) | t2_(B*i+2));
    tensor_out(B*i+3) = cutlass::half_t::bitcast((t1_(B*i+3) << 8) | t2_(B*i+3));
  }
  */

  /*
  // Unrolling, no effect

  int B = 2;

  CUTE_UNROLL
  for (int i = 0; i < size(tensor_in1); i += B) {
    int16_t a1_ = static_cast<int16_t>(tensor_in1(i).raw()) << 8;
    uint16_t a1 = static_cast<uint16_t>(a1_ >> 1);
    uint16_t b1 = tensor_in2(i).raw();
    uint16_t sub1 = (a1 ^ b1) & 0x80;
    tensor_out(i) = cutlass::half_t::bitcast((a1 - sub1) & 0xbf00 | b1);

    int16_t a2_ = static_cast<int16_t>(tensor_in1(i+1).raw()) << 8;
    uint16_t a2 = static_cast<uint16_t>(a2_ >> 1);
    uint16_t b2 = tensor_in2(i+1).raw();
    uint16_t sub2 = (a2 ^ b2) & 0x80;
    tensor_out(i+1) = cutlass::half_t::bitcast((a2 - sub2) & 0xbf00 | b2);
  }
  */

  /*
  // Baseline 

  CUTE_UNROLL
  for (int i = 0; i < size(tensor_in1); i++) {
    uint8_t a = tensor_in1(i).raw();
    uint8_t b = tensor_in2(i).raw();
    uint8_t s = a & 0x80;
    uint8_t sub = (a & 0x1) ^ (b >> 7);
    uint8_t e_ = ((a - sub) >> 1) & 0x3f;
  
    tensor_out(i) = cutlass::half_t::bitcast(((s | e_) << 8) | b);
  }
  */
}

// Accept mutable temporaries
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut>
CUTE_DEVICE constexpr
void
transform2(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>     && tensor_out)
{
  return transform2(tensor_in1, tensor_in2, tensor_out);
}

template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut>
CUTE_DEVICE constexpr
void
transform3(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>      & tensor_out)
{
  // Batch logic operations
  int B = 4;
  int SZ = size(tensor_in1) * 2;

  Tensor t1 = recast<uint32_t>(flatten(tensor_in1));
  Tensor t2 = recast<uint32_t>(flatten(tensor_in2));
  Tensor t3 = recast<uint32_t>(flatten(tensor_out));

  CUTE_UNROLL
  for (int i = 0; i < SZ / B; i++) {
    uint32_t a = t1(i);
    uint32_t b = t2(i);
    uint32_t s = a & 0x80808080;
    uint32_t sub = (b & 0x80808080) >> 7;
    t1(i) = (((a - sub) >> 1) & 0x3f3f3f3f) | s;
    uint32_t c = __byte_perm(t1(i), b, 0x1504);
    uint32_t d = __byte_perm(t1(i), b, 0x3726);
    t3((B/2)*i) = c;
    t3((B/2)*i+1) = d;
  }
}

// Accept mutable temporaries
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut>
CUTE_DEVICE constexpr
void
transform3(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>     && tensor_out)
{
  return transform3(tensor_in1, tensor_in2, tensor_out);
}

template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut>
CUTE_DEVICE constexpr
void
transform4(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>      & tensor_out)
{
  CUTE_UNROLL
  for (int i = 0; i < size(tensor_in1); i++) {
    uint8_t a = tensor_in1(i).raw();
    uint8_t b = tensor_in2(i).raw();
    uint8_t s = a & 0x80;
    uint8_t sub = (a & 0x1) ^ (b >> 7);
    uint8_t e_ = ((a - sub) >> 1) & 0x3f;
  
    tensor_out(i) = cutlass::half_t::bitcast(((s | e_) << 8) | b);
  }
}

// Accept mutable temporaries
template <class EngineIn1, class LayoutIn1,
          class EngineIn2, class LayoutIn2,
          class EngineOut, class LayoutOut>
CUTE_DEVICE constexpr
void
transform4(Tensor<EngineIn1,LayoutIn1> const& tensor_in1,
          Tensor<EngineIn2,LayoutIn2> const& tensor_in2,
          Tensor<EngineOut,LayoutOut>     && tensor_out)
{
  return transform4(tensor_in1, tensor_in2, tensor_out);
}

} // end namespace cute
