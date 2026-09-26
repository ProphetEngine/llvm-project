//===----- hlsl_patch.h - HLSL tessellation patch types ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_PATCH_H_
#define _HLSL_HLSL_PATCH_H_

namespace hlsl {

// The stage entry wrapper supplies the control points. Shader code may only
// read them through the patch parameter.
template <typename T, unsigned N> class InputPatch {
  T Points[N];

public:
  T operator[](unsigned Index) const { return Points[Index]; }
};

template <typename T, unsigned N> class OutputPatch {
  T Points[N];

public:
  T operator[](unsigned Index) const { return Points[Index]; }
};

} // namespace hlsl

#endif // _HLSL_HLSL_PATCH_H_
