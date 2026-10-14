//===----- hlsl_spirv.h - HLSL definitions for SPIR-V target --------------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef _HLSL_HLSL_SPIRV_H_
#define _HLSL_HLSL_SPIRV_H_

namespace hlsl {
namespace vk {
template <typename T, T v> struct integral_constant {
  static constexpr T value = v;
};

template <typename T> struct Literal {};

template <uint Opcode, uint Size, uint Alignment, typename... Operands>
using SpirvType = __hlsl_spirv_type<Opcode, Size, Alignment, Operands...>;

template <uint Opcode, typename... Operands>
using SpirvOpaqueType = __hlsl_spirv_type<Opcode, 0, 0, Operands...>;
} // namespace vk
} // namespace hlsl

// SPV_KHR_ray_tracing exposes the zero-based launch coordinates as a uint3.
inline uint3 DispatchRaysIndex() {
  return uint3(__builtin_spirv_launch_id(0),
               __builtin_spirv_launch_id(1),
               __builtin_spirv_launch_id(2));
}

// Dispatching tasks terminates the current amplification workgroup. The
// payload is checked by Sema and lowered to TaskPayloadWorkgroupEXT.
#ifdef __PROPHET_SHADER_CPP__
template <typename T>
void DispatchMesh(uint x, uint y, uint z, groupshared T &payload);
#else
__attribute__((clang_builtin_alias(__builtin_hlsl_dispatch_mesh)))
void DispatchMesh(uint x, uint y, uint z, ...);
#endif

// Intrinsic declarations used by the typed AST backend. The implementation is
// emitted as SPIR-V, including the required resource address for atomics.
bool isfinite(float value);
bool2 isfinite(float2 value);
bool3 isfinite(float3 value);
bool4 isfinite(float4 value);
bool isfinite(double value);
bool2 isfinite(double2 value);
bool3 isfinite(double3 value);
bool4 isfinite(double4 value);
template<typename T>
void InterlockedCompareExchange(inout T destination, T compare, T value, out T original);
template<typename T>
void InterlockedCompareStore(inout T destination, T compare, T value);

#endif // _HLSL_HLSL_SPIRV_H_
