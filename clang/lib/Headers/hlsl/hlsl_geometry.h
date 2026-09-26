//===----- hlsl_geometry.h - HLSL geometry stream types ------------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_GEOMETRY_H_
#define _HLSL_HLSL_GEOMETRY_H_

namespace hlsl {

// Geometry output streams are supplied by the graphics pipeline. Their
// operations are shader builtins and require stage-specific code generation.
template <typename T> class PointStream {
public:
  void Append(T value);
  void RestartStrip();
};

template <typename T> class LineStream {
public:
  void Append(T value);
  void RestartStrip();
};

template <typename T> class TriangleStream {
public:
  void Append(T value);
  void RestartStrip();
};

} // namespace hlsl

#endif // _HLSL_HLSL_GEOMETRY_H_
