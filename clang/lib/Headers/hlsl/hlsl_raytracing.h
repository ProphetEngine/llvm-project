//===-- hlsl_raytracing.h - Shader ray pipeline declarations --------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef _HLSL_RAYTRACING_H_
#define _HLSL_RAYTRACING_H_

// These declarations are consumed by the shared AST SPIR-V emitter.
struct RaytracingAccelerationStructure {
  __hlsl_resource_t [[hlsl::resource_class("SRV")]] __handle;
};
struct RayDesc {
  float3 Origin;
  float TMin;
  float3 Direction;
  float TMax;
};
struct BuiltInTriangleIntersectionAttributes { float2 barycentrics; };

static const uint RAY_FLAG_NONE = 0;
static const uint RAY_FLAG_FORCE_OPAQUE = 0x1;
static const uint RAY_FLAG_FORCE_NON_OPAQUE = 0x2;
static const uint RAY_FLAG_ACCEPT_FIRST_HIT_AND_END_SEARCH = 0x4;
static const uint RAY_FLAG_SKIP_CLOSEST_HIT_SHADER = 0x8;
static const uint RAY_FLAG_CULL_BACK_FACING_TRIANGLES = 0x10;
static const uint RAY_FLAG_CULL_FRONT_FACING_TRIANGLES = 0x20;
static const uint RAY_FLAG_CULL_OPAQUE = 0x40;
static const uint RAY_FLAG_CULL_NON_OPAQUE = 0x80;
static const uint RAY_FLAG_SKIP_TRIANGLES = 0x100;
static const uint RAY_FLAG_SKIP_PROCEDURAL_PRIMITIVES = 0x200;
static const uint HIT_KIND_TRIANGLE_FRONT_FACE = 0xfe;
static const uint HIT_KIND_TRIANGLE_BACK_FACE = 0xff;

static const uint CANDIDATE_NON_OPAQUE_TRIANGLE = 0;
static const uint CANDIDATE_PROCEDURAL_PRIMITIVE = 1;
static const uint COMMITTED_NOTHING = 0;
static const uint COMMITTED_TRIANGLE_HIT = 1;
static const uint COMMITTED_PROCEDURAL_PRIMITIVE_HIT = 2;

// Opaque function-local state. The AST emitter lowers these declarations to
// RayQueryKHR instructions; no ordinary record storage or copy is permitted.
template<uint Flags> struct RayQuery {
  void TraceRayInline(RaytracingAccelerationStructure scene, uint flags,
                      uint inclusionMask, RayDesc ray);
  bool Proceed();
  void Abort();
  void CommitNonOpaqueTriangleHit();
  void CommitProceduralPrimitiveHit(float rayT);
  uint CandidateType();
  uint CommittedStatus();
  uint RayFlags();
  float RayTMin();
  float3 WorldRayOrigin();
  float3 WorldRayDirection();
  float CandidateTriangleRayT();
  float CommittedRayT();
  uint CandidateInstanceID();
  uint CommittedInstanceID();
  uint CandidateInstanceIndex();
  uint CommittedInstanceIndex();
  uint CandidateGeometryIndex();
  uint CommittedGeometryIndex();
  uint CandidatePrimitiveIndex();
  uint CommittedPrimitiveIndex();
  float2 CandidateTriangleBarycentrics();
  float2 CommittedTriangleBarycentrics();
  bool CandidateTriangleFrontFace();
  bool CommittedTriangleFrontFace();
  bool CandidateProceduralPrimitiveNonOpaque();
  float3 CandidateObjectRayOrigin();
  float3 CommittedObjectRayOrigin();
  float3 CandidateObjectRayDirection();
  float3 CommittedObjectRayDirection();
  float3x4 CandidateObjectToWorld3x4();
  float3x4 CommittedObjectToWorld3x4();
  float3x4 CandidateWorldToObject3x4();
  float3x4 CommittedWorldToObject3x4();
};

template<typename T>
void TraceRay(RaytracingAccelerationStructure scene, uint rayFlags, uint inclusionMask,
              uint rayContributionToHitGroupIndex, uint multiplierForGeometryContributionToHitGroupIndex,
              uint missShaderIndex, RayDesc ray, inout T payload);
template<typename T> void CallShader(uint shaderIndex, inout T data);
template<typename T> bool ReportHit(float hitT, uint hitKind, T attributes);
void IgnoreHit();
void AcceptHitAndEndSearch();
uint3 DispatchRaysDimensions();
float3 WorldRayOrigin();
float3 WorldRayDirection();
float3 ObjectRayOrigin();
float3 ObjectRayDirection();
float RayTMin();
float RayTCurrent();
uint RayFlags();
uint InstanceIndex();
uint InstanceID();
uint GeometryIndex();
uint PrimitiveIndex();
uint HitKind();
float3x4 ObjectToWorld3x4();
float3x4 WorldToObject3x4();
float4x3 ObjectToWorld4x3();
float4x3 WorldToObject4x3();

#endif
