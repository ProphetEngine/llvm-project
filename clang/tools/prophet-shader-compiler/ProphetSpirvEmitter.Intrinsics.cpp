//===-- ProphetSpirvEmitter.Intrinsics.cpp - Common shader operations ------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

SpirvValue ProphetSpirvEmitter::intrinsic(StringRef Name, const CallExpr *E,
    ArrayRef<SpirvValue> Args, bool &Handled) {
  Handled = true;
  Name.consume_front("__builtin_elementwise_");
  Name.consume_front("__builtin_hlsl_");
  uint32_t T = E->getType()->isVoidType() ? 0 : intrinsicType(E->getType());
  auto Result = [&](uint32_t V) { return convert({V, T}, type(E->getType()), E->getType()); };
  auto Arg = [&](unsigned I) {
    SpirvValue V = load(Args[I]);
    if (!V.ASTType.isNull() && !V.Resource) V = convert(V, intrinsicType(V.ASTType), V.ASTType);
    return V;
  };
  auto ID = [&](unsigned I) { return Arg(I).Id; };
  auto Scope = [&](uint32_t S) { return constant(integerType(), S); };
  std::vector<uint32_t> IDs;

  uint32_t BI = StringSwitch<uint32_t>(Name)
      .Case("DispatchRaysIndex", BuiltIn::LaunchIdKHR).Case("DispatchRaysDimensions", BuiltIn::LaunchSizeKHR)
      .Case("WorldRayOrigin", BuiltIn::WorldRayOriginKHR).Case("WorldRayDirection", BuiltIn::WorldRayDirectionKHR)
      .Case("ObjectRayOrigin", BuiltIn::ObjectRayOriginKHR).Case("ObjectRayDirection", BuiltIn::ObjectRayDirectionKHR)
      .Case("RayTMin", BuiltIn::RayTminKHR).Case("RayTCurrent", BuiltIn::RayTmaxKHR)
      .Case("InstanceIndex", BuiltIn::InstanceId).Case("InstanceID", BuiltIn::InstanceCustomIndexKHR)
      .Case("GeometryIndex", BuiltIn::RayGeometryIndexKHR).Case("PrimitiveIndex", BuiltIn::PrimitiveId)
      .Case("HitKind", BuiltIn::HitKindKHR).Case("RayFlags", BuiltIn::IncomingRayFlagsKHR)
      .Cases({"ObjectToWorld3x4", "ObjectToWorld4x3"}, BuiltIn::ObjectToWorldKHR)
      .Cases({"WorldToObject3x4", "WorldToObject4x3"}, BuiltIn::WorldToObjectKHR)
      .Case("WaveGetLaneIndex", BuiltIn::SubgroupLocalInvocationId)
      .Case("WaveGetLaneCount", BuiltIn::SubgroupSize).Default(~0u);
  if (BI != ~0u) {
    if (Name.starts_with("Wave")) capability(Capability::GroupNonUniform);
    else if (CurrentStage < 10) { fail("ray intrinsic requires a ray shader stage", E->getExprLoc()); return {}; }
    uint32_t BuiltinType = T;
    if (Name.ends_with("4x3")) BuiltinType = type(Ctx.getConstantMatrixType(Ctx.FloatTy, 3, 4));
    uint32_t Value = load({builtin(BI, BuiltinType), BuiltinType, StorageClass::Input, E->getType(), true}).Id;
    if (BuiltinType != T) Value = valueOp(OpTranspose, T, {Value});
    return Result(Value);
  }
  if (Name == "GroupMemoryBarrierWithGroupSync" || Name == "AllMemoryBarrierWithGroupSync" ||
      Name == "DeviceMemoryBarrierWithGroupSync" || Name == "GroupMemoryBarrier" ||
      Name == "DeviceMemoryBarrier" || Name == "AllMemoryBarrier") {
    if (Name.ends_with("WithGroupSync")) {
      if (CurrentStage != 1 && CurrentStage != 7 && CurrentStage != 8 && !(ObjectMode && CurrentStage == 0)) {
        fail("workgroup barrier requires compute, task or mesh stage", E->getExprLoc()); return {};
      }
      trackBarrier(E->getExprLoc());
    }
    uint32_t Memory = MemorySemantics::AcquireRelease;
    if (!Name.starts_with("Device")) Memory |= MemorySemantics::WorkgroupMemory;
    if (!Name.starts_with("Group")) Memory |= MemorySemantics::UniformMemory | MemorySemantics::ImageMemory;
    uint32_t MemoryScope = Name.starts_with("Group") ? spv::Scope::Workgroup : spv::Scope::Device;
    if (Name.ends_with("WithGroupSync"))
      Body.emit(OpControlBarrier, {Scope(spv::Scope::Workgroup), Scope(MemoryScope), Scope(Memory)});
    else Body.emit(OpMemoryBarrier, {Scope(MemoryScope), Scope(Memory)});
    return {};
  }
  if (Name.starts_with("Interlocked")) {
    if (Args.size() < 2) { fail("atomic operation requires a target and value", E->getExprLoc()); return {}; }
    SpirvValue Target = address(Args[0]);
    if (Target.Swizzle.size() == 1 && Types[Target.Type].K == SpirvType::Vector) {
      uint32_t Component = Target.Swizzle.front(), Element = Types[Target.Type].Element;
      Target.Swizzle.clear();
      Target = access(Target, {constant(integerType(), Component)}, Target.ASTType, Element);
    }
    if (!Target.Address) { fail("atomic target must be a shader lvalue", E->getExprLoc()); return {}; }
    if (Target.Storage != StorageClass::StorageBuffer && Target.Storage != StorageClass::Workgroup) {
      fail("atomic target must be a storage buffer or groupshared lvalue", E->getExprLoc()); return {};
    }
    if (Types[Target.Type].K != SpirvType::Int || (Types[Target.Type].Bits != 32 && Types[Target.Type].Bits != 64)) {
      fail("atomic target requires a 32-bit or 64-bit integer", E->getExprLoc()); return {};
    }
    if (Types[Target.Type].Bits == 64) capability(Capability::Int64Atomics);
    uint32_t MemoryScope = Target.Storage == StorageClass::Workgroup ? spv::Scope::Workgroup : spv::Scope::Device;
    bool Signed = Types[Target.Type].Signed;
    uint32_t Op = StringSwitch<uint32_t>(Name)
        .Case("InterlockedAdd", OpAtomicIAdd).Case("InterlockedAnd", OpAtomicAnd)
        .Case("InterlockedOr", OpAtomicOr).Case("InterlockedXor", OpAtomicXor)
        .Case("InterlockedMin", Signed ? OpAtomicSMin : OpAtomicUMin)
        .Case("InterlockedMax", Signed ? OpAtomicSMax : OpAtomicUMax)
        .Case("InterlockedExchange", OpAtomicExchange)
        .Cases({"InterlockedCompareExchange", "InterlockedCompareStore"}, OpAtomicCompareExchange).Default(0);
    if (!Op) { fail("unsupported atomic intrinsic: " + Name, E->getExprLoc()); return {}; }
    std::vector<uint32_t> Operands{Target.Id, Scope(MemoryScope), Scope(0)};
    unsigned Original = 2;
    if (Op == OpAtomicCompareExchange) {
      if (Args.size() < 3) { fail("compare exchange requires compare and replacement values", E->getExprLoc()); return {}; }
      Operands.push_back(Scope(0)); Operands.push_back(ID(2)); Operands.push_back(ID(1)); Original = 3;
    } else Operands.push_back(ID(1));
    uint32_t Previous = valueOp(Op, Target.Type, Operands);
    if (Args.size() > Original) store(Args[Original], {Previous, Target.Type});
    return T ? Result(Previous) : SpirvValue{};
  }
  if (Name.starts_with("Wave") || Name.starts_with("Quad")) {
    capability(Capability::GroupNonUniform);
    uint32_t S = Scope(spv::Scope::Subgroup), Op = 0;
    if (Name == "WaveIsFirstLane") {
      return Result(valueOp(OpGroupNonUniformElect, T, {S}));
    }
    if (Name == "WaveReadLaneFirst" || Name == "WaveReadLaneAt") {
      bool First = Name == "WaveReadLaneFirst";
      capability(First ? Capability::GroupNonUniformBallot : Capability::GroupNonUniformShuffle);
      std::vector<uint32_t> Operands{S, ID(0)}; if (!First) Operands.push_back(ID(1));
      return Result(valueOp(First ? OpGroupNonUniformBroadcastFirst : OpGroupNonUniformShuffle, T, Operands));
    }
    if (Name == "WaveActiveAnyTrue" || Name == "WaveActiveAllTrue" || Name == "WaveActiveAllEqual") {
      capability(Capability::GroupNonUniformVote);
      Op = Name == "WaveActiveAnyTrue" ? OpGroupNonUniformAny :
           Name == "WaveActiveAllTrue" ? OpGroupNonUniformAll : OpGroupNonUniformAllEqual;
      return Result(valueOp(Op, T, {S, ID(0)}));
    }
    if (Name == "WaveActiveBallot" || Name == "WaveActiveCountBits" || Name == "WavePrefixCountBits") {
      capability(Capability::GroupNonUniformBallot);
      uint32_t Mask = valueOp(OpGroupNonUniformBallot, vectorType(integerType(), 4),
          {S, convert(Args[0], boolType()).Id});
      if (Name == "WaveActiveBallot") return Result(Mask);
      uint32_t Group = Name == "WavePrefixCountBits" ? GroupOperation::ExclusiveScan : GroupOperation::Reduce;
      return Result(valueOp(OpGroupNonUniformBallotBitCount, T, {S, Group, Mask}));
    }
    if (Name.starts_with("Quad")) {
      capability(Capability::GroupNonUniformQuad);
      if (Name == "QuadReadLaneAt") return Result(valueOp(OpGroupNonUniformQuadBroadcast, T, {S, ID(0), ID(1)}));
      uint32_t Direction = Name == "QuadReadAcrossX" ? 0 : Name == "QuadReadAcrossY" ? 1 : 2;
      return Result(valueOp(OpGroupNonUniformQuadSwap, T, {S, ID(0), constant(integerType(), Direction)}));
    }
    capability(Capability::GroupNonUniformArithmetic);
    SpirvType Element = Types[T]; if (Element.K == SpirvType::Vector) Element = Types[Element.Element];
    bool Float = Element.K == SpirvType::Float, Signed = Element.Signed;
    if (Name.ends_with("Sum")) Op = Float ? OpGroupNonUniformFAdd : OpGroupNonUniformIAdd;
    if (Name.ends_with("Product")) Op = Float ? OpGroupNonUniformFMul : OpGroupNonUniformIMul;
    if (Name.ends_with("Min")) Op = Float ? OpGroupNonUniformFMin : Signed ? OpGroupNonUniformSMin : OpGroupNonUniformUMin;
    if (Name.ends_with("Max")) Op = Float ? OpGroupNonUniformFMax : Signed ? OpGroupNonUniformSMax : OpGroupNonUniformUMax;
    if (Name.ends_with("BitAnd")) Op = OpGroupNonUniformBitwiseAnd;
    if (Name.ends_with("BitOr")) Op = OpGroupNonUniformBitwiseOr;
    if (Name.ends_with("BitXor")) Op = OpGroupNonUniformBitwiseXor;
    if (Op) return Result(valueOp(Op, T, {S, Name.starts_with("WavePrefix") ? GroupOperation::ExclusiveScan : GroupOperation::Reduce, ID(0)}));
    fail("unsupported wave intrinsic: " + Name, E->getExprLoc()); return {};
  }
  if (Name == "SetMeshOutputCounts") {
    if (CurrentStage != 8) { fail("SetMeshOutputCounts requires a mesh entry", E->getExprLoc()); return {}; }
    Body.emit(OpSetMeshOutputsEXT, {ID(0), ID(1)}); return {};
  }
  if (Name == "DispatchMesh") {
    if (CurrentStage != 7 || Args.size() != 4) { fail("DispatchMesh requires task stage and payload", E->getExprLoc()); return {}; }
    trackBarrier(E->getExprLoc());
    uint32_t Memory = MemorySemantics::AcquireRelease | MemorySemantics::WorkgroupMemory;
    Body.emit(OpControlBarrier, {Scope(spv::Scope::Workgroup), Scope(spv::Scope::Workgroup), Scope(Memory)});
    if (!Args[3].Address || (Args[3].Storage != StorageClass::Workgroup &&
                            Args[3].Storage != StorageClass::TaskPayloadWorkgroupEXT)) {
      fail("DispatchMesh requires a global groupshared record payload", E->getExprLoc()); return {};
    }
    uint32_t PayloadType = type(Args[3].ASTType);
    if (Args[3].Storage == StorageClass::TaskPayloadWorkgroupEXT) {
      uint32_t Variable = pointerRoot(Args[3].Id);
      UsedGlobals.insert(Variable);
      Body.emit(OpEmitMeshTasksEXT, {ID(0), ID(1), ID(2), Variable});
      Terminated = true; return {};
    }
    if (!MeshPayload) {
      uint32_t V = globalVariable(PayloadType, StorageClass::TaskPayloadWorkgroupEXT, "task.payload");
      MeshPayload = {V, PayloadType, StorageClass::TaskPayloadWorkgroupEXT, Args[3].ASTType, true};
    }
    if (MeshPayload.Type != PayloadType) {
      fail("all DispatchMesh sites must use the same payload type", E->getExprLoc()); return {};
    }
    uint32_t Variable = MeshPayload.Id;
    UsedGlobals.insert(Variable);
    uint32_t Lane = load({builtin(BuiltIn::LocalInvocationIndex, integerType()), integerType(), StorageClass::Input, Ctx.UnsignedIntTy, true}).Id;
    uint32_t First = valueOp(OpIEqual, boolType(), {Lane, constant(integerType(), 0)});
    uint32_t Copy = id(), Merge = id(); Body.emit(OpSelectionMerge, {Merge, 0});
    conditionalBranch(First, Copy, Merge); label(Copy);
    store({Variable, PayloadType, StorageClass::TaskPayloadWorkgroupEXT, Args[3].ASTType, true}, Args[3]);
    branch(Merge); label(Merge);
    Body.emit(OpControlBarrier, {Scope(spv::Scope::Workgroup), Scope(spv::Scope::Workgroup), Scope(Memory)});
    Body.emit(OpEmitMeshTasksEXT, {ID(0), ID(1), ID(2), Variable}); Terminated = true; return {};
  }
  if (Name == "TraceRay") {
    if (CurrentStage < 10 || Args.size() != 8) { fail("TraceRay requires a ray stage and eight arguments", E->getExprLoc()); return {}; }
    SpirvValue Payload = Arg(7), Ray = Arg(6);
    SpirvValue Pointer = raySlot(Payload, StorageClass::RayPayloadKHR, "ray.payload");
    if (!Pointer) return {}; uint32_t Variable = Pointer.Id; store(Pointer, Payload);
    auto Field = [&](uint32_t I) {
      uint32_t Member = Types[Ray.Type].Members[I];
      SpirvValue V{valueOp(OpCompositeExtract, Member, {Ray.Id, I}), Member};
      return convert(V, I == 0 || I == 2 ? vectorType(floatType(), 3) : floatType()).Id;
    };
    Body.emit(OpTraceRayKHR, {ID(0), ID(1), ID(2), ID(3), ID(4), ID(5), Field(0), Field(1), Field(2), Field(3), Variable});
    store(Args[7], Pointer); return {};
  }
  if (Name == "CallShader") {
    if (CurrentStage < 10 || Args.size() != 2) { fail("CallShader requires a ray stage and callable data", E->getExprLoc()); return {}; }
    SpirvValue Data = Arg(1);
    SpirvValue Pointer = raySlot(Data, StorageClass::CallableDataKHR, "callable.data");
    if (!Pointer) return {}; uint32_t Variable = Pointer.Id; store(Pointer, Data);
    Body.emit(OpExecuteCallableKHR, {ID(0), Variable}); store(Args[1], Pointer); return {};
  }
  if (Name == "ReportHit") {
    if (CurrentStage != 11 || Args.size() != 3) { fail("ReportHit requires an intersection shader", E->getExprLoc()); return {}; }
    SpirvValue Data = Arg(2);
    SpirvValue Pointer = raySlot(Data, StorageClass::HitAttributeKHR, "hit.attribute");
    if (!Pointer) return {}; store(Pointer, Data);
    return Result(valueOp(OpReportIntersectionKHR, T, {ID(0), ID(1)}));
  }
  if (Name == "IgnoreHit" || Name == "AcceptHitAndEndSearch") {
    if (CurrentStage != 12) { fail("hit control intrinsic requires anyhit stage", E->getExprLoc()); return {}; }
    Body.emit(Name == "IgnoreHit" ? OpIgnoreIntersectionKHR : OpTerminateRayKHR); Terminated = true; return {};
  }
  if (Name == "clip" || Name == "discard") {
    if (CurrentStage != 3 && !(ObjectMode && CurrentStage == 0)) { fail("discard requires a pixel entry", E->getExprLoc()); return {}; }
    if (Args.empty()) { Body.emit(OpKill); Terminated = true; return {}; }
    SpirvValue V = Arg(0); SpirvType Info = Types[V.Type];
    uint32_t BT = Info.K == SpirvType::Vector ? vectorType(boolType(), Info.Count) : boolType();
    uint32_t C = valueOp(OpFOrdLessThan, BT, {V.Id, zero(V.Type)});
    if (Info.K == SpirvType::Vector) C = valueOp(OpAny, boolType(), {C});
    uint32_t Kill = id(), Merge = id(); Body.emit(OpSelectionMerge, {Merge, 0});
    conditionalBranch(C, Kill, Merge); label(Kill); Body.emit(OpKill); Terminated = true; label(Merge); return {};
  }
  if (Name == "ddx" || Name == "ddy" || Name == "fwidth" || Name == "ddx_coarse" ||
      Name == "ddy_coarse" || Name == "ddx_fine" || Name == "ddy_fine") {
    if (CurrentStage != 3 && !(ObjectMode && CurrentStage == 0)) { fail("pixel derivatives require a fragment entry", E->getExprLoc()); return {}; }
    uint32_t Op = StringSwitch<uint32_t>(Name).Case("ddx", OpDPdx).Case("ddy", OpDPdy).Case("fwidth", OpFwidth)
        .Case("ddx_coarse", OpDPdxCoarse).Case("ddy_coarse", OpDPdyCoarse).Case("ddx_fine", OpDPdxFine).Case("ddy_fine", OpDPdyFine).Default(0);
    if (Name.contains('_')) capability(Capability::DerivativeControl);
    return Result(valueOp(Op, T, {ID(0)}));
  }
  if (Name == "asuint" || Name == "asint" || Name == "asfloat" || Name == "asdouble") {
    if (Args.size() == 1) return Result(valueOp(OpBitcast, T, {ID(0)}));
    if (Name == "asdouble" && Args.size() == 2) {
      uint32_t Pair = composite(vectorType(integerType(), 2), {ID(0), ID(1)});
      return Result(valueOp(OpBitcast, T, {Pair}));
    }
  }
  if (Name == "countbits" || Name == "reversebits")
    return Result(valueOp(Name == "countbits" ? OpBitCount : OpBitReverse, T, {ID(0)}));
  if (Name == "isnan" || Name == "isinf")
    return Result(valueOp(Name == "isnan" ? OpIsNan : OpIsInf, T, {ID(0)}));
  if (Name == "isfinite") {
    uint32_t Value = ID(0), Nan = valueOp(OpIsNan, T, {Value}), Inf = valueOp(OpIsInf, T, {Value});
    return Result(valueOp(OpLogicalNot, T, {valueOp(OpLogicalOr, T, {Nan, Inf})}));
  }
  if (Name == "rcp") {
    uint32_t Element = Types[T].K == SpirvType::Vector ? Types[T].Element : T;
    uint64_t Bits = Types[Element].Bits == 64 ? 0x3ff0000000000000ull : 0x3f800000u;
    SpirvValue One = convert({constant(Element, Bits), Element}, T);
    return Result(valueOp(OpFDiv, T, {One.Id, ID(0)}));
  }
  if (Name == "f16tof32" || Name == "f32tof16") {
    SpirvValue Input = Arg(0);
    uint32_t Count = Types[T].K == SpirvType::Vector ? Types[T].Count : 1;
    uint32_t InElement = Types[Input.Type].K == SpirvType::Vector ? Types[Input.Type].Element : Input.Type;
    uint32_t F32 = floatType(), U32 = integerType(), F2 = vectorType(F32, 2);
    std::vector<uint32_t> Values;
    for (uint32_t I = 0; I < Count; ++I) {
      uint32_t V = Count == 1 ? Input.Id : valueOp(OpCompositeExtract, InElement, {Input.Id, I});
      if (Name == "f16tof32") {
        uint32_t Pair = extOp(GLSL::UnpackHalf2x16, F2, {V});
        Values.push_back(valueOp(OpCompositeExtract, F32, {Pair, 0}));
      } else {
        uint32_t Pair = composite(F2, {V, zero(F32)});
        Values.push_back(extOp(GLSL::PackHalf2x16, U32, {Pair}));
      }
    }
    return Result(Count == 1 ? Values.front() : composite(T, Values));
  }
  if (Name == "any" || Name == "all") {
    SpirvValue V = Arg(0); uint32_t Count = Types[V.Type].Count;
    V = convert(V, Count ? vectorType(boolType(), Count) : boolType());
    return Result(Count ? valueOp(Name == "any" ? OpAny : OpAll, T, {V.Id}) : V.Id);
  }
  if (Name == "dot") return Result(valueOp(OpDot, T, {ID(0), ID(1)}));
  if (Name == "sign") {
    // HLSL returns int/intN even for floating operands. Selecting GLSL SSign
    // from the result type is invalid; ordered input comparisons also retain
    // HLSL's zero result for signed zero and unordered (NaN) operands.
    SpirvValue V = Arg(0);
    SpirvType Input = Types[V.Type];
    uint32_t BT = Input.K == SpirvType::Vector ? vectorType(boolType(), Input.Count) : boolType();
    SpirvType Scalar = Input.K == SpirvType::Vector ? Types[Input.Element] : Input;
    uint32_t Element = Types[T].K == SpirvType::Vector ? Types[T].Element : T;
    SpirvValue One = convert({constant(integerType(), 1), integerType()}, T);
    uint32_t Positive = valueOp(Scalar.K == SpirvType::Float ? OpFOrdGreaterThan : Scalar.Signed ? OpSGreaterThan : OpUGreaterThan,
                                BT, {V.Id, zero(V.Type)});
    uint32_t P = valueOp(OpSelect, T, {Positive, One.Id, zero(T)});
    if (Scalar.K != SpirvType::Float && !Scalar.Signed) return Result(P);
    uint32_t Negative = valueOp(Scalar.K == SpirvType::Float ? OpFOrdLessThan : OpSLessThan, BT, {V.Id, zero(V.Type)});
    uint32_t N = valueOp(OpSelect, T, {Negative, One.Id, zero(T)});
    return Result(valueOp(Types[Element].K == SpirvType::Float ? OpFSub : OpISub, T, {P, N}));
  }
  if (Name == "transpose") return matrixTranspose(Args[0], E->getType());
  if (Name == "mul") return matrixMultiply(Args[0], Args[1], E->getType());
  SpirvType Scalar = Types[T]; if (Scalar.K == SpirvType::Vector) Scalar = Types[Scalar.Element];
  bool Float = Scalar.K == SpirvType::Float, Signed = Scalar.Signed;
  uint32_t Ext = StringSwitch<uint32_t>(Name)
      .Case("round", GLSL::RoundEven).Case("trunc", GLSL::Trunc)
      .Case("abs", Float ? GLSL::FAbs : GLSL::SAbs)
      .Case("floor", GLSL::Floor).Case("ceil", GLSL::Ceil).Case("frac", GLSL::Fract)
      .Case("radians", GLSL::Radians).Case("degrees", GLSL::Degrees)
      .Case("sin", GLSL::Sin).Case("cos", GLSL::Cos).Case("tan", GLSL::Tan)
      .Case("asin", GLSL::Asin).Case("acos", GLSL::Acos).Case("atan", GLSL::Atan).Case("atan2", GLSL::Atan2)
      .Case("sinh", GLSL::Sinh).Case("cosh", GLSL::Cosh).Case("tanh", GLSL::Tanh)
      .Case("pow", GLSL::Pow).Case("exp", GLSL::Exp).Case("exp2", GLSL::Exp2)
      .Case("log", GLSL::Log).Case("log2", GLSL::Log2).Case("sqrt", GLSL::Sqrt).Case("rsqrt", GLSL::InverseSqrt)
      .Case("determinant", GLSL::Determinant)
      .Case("min", Float ? GLSL::FMin : Signed ? GLSL::SMin : GLSL::UMin)
      .Case("max", Float ? GLSL::FMax : Signed ? GLSL::SMax : GLSL::UMax)
      .Case("clamp", Float ? GLSL::FClamp : Signed ? GLSL::SClamp : GLSL::UClamp)
      .Case("lerp", GLSL::FMix).Case("step", GLSL::Step).Case("smoothstep", GLSL::SmoothStep)
      .Cases({"mad", "fma"}, GLSL::Fma).Case("length", GLSL::Length).Case("distance", GLSL::Distance)
      .Case("cross", GLSL::Cross).Case("normalize", GLSL::Normalize).Case("reflect", GLSL::Reflect)
      .Case("refract", GLSL::Refract).Case("faceforward", GLSL::FaceForward)
      .Case("firstbitlow", GLSL::FindILsb).Case("firstbithigh", Signed ? GLSL::FindSMsb : GLSL::FindUMsb)
      .Default(0);
  if (Ext) {
    for (unsigned I = 0; I < Args.size(); ++I) {
      SpirvValue Value = Arg(I);
      // HLSL permits scalar factors/bounds for componentwise vector intrinsics.
      // GLSL.std.450 requires matching vector operands (except refract's eta).
      if (Types[T].K == SpirvType::Vector && Types[Value.Type].K != SpirvType::Vector &&
          !(Ext == GLSL::Refract && I == 2))
        Value = convert(Value, T);
      IDs.push_back(Value.Id);
    }
    return Result(extOp(Ext, T, IDs));
  }
  if (Name == "saturate") {
    uint32_t Element = Types[T].K == SpirvType::Vector ? Types[T].Element : T;
    SpirvValue One = convert({constant(Element, Scalar.Bits == 64 ? 0x3ff0000000000000ull : 0x3f800000u), Element}, T);
    return Result(extOp(GLSL::FClamp, T, {ID(0), zero(T), One.Id}));
  }
  Handled = false; return {};
}
}
