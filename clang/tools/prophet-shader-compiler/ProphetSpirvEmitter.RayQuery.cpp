// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

bool ProphetSpirvEmitter::isRayQuery(QualType T) const {
  if (T.isNull()) return false;
  const auto *R = T.getNonReferenceType()->getAsCXXRecordDecl();
  return R && R->getName() == "RayQuery" &&
      (R->isHLSLBuiltinRecord() || Ctx.getSourceManager().isInSystemHeader(R->getLocation()));
}

SpirvValue ProphetSpirvEmitter::rayQueryCall(StringRef Name, SpirvValue Object,
    const CallExpr *E, ArrayRef<SpirvValue> Args) {
  if (!Object.Address || Object.Storage != StorageClass::Function || !Object.Path.empty()) {
    fail("RayQuery operations require a function-local query variable", E->getExprLoc()); return {};
  }
  if (Environment == 1) {
    fail("RayQuery requires Vulkan 1.2 or later", E->getExprLoc()); return {};
  }
  uint32_t Query = Object.Id;
  if (Name == "TraceRayInline") {
    if (Args.size() != 4 || !Args[0].Resource ||
        Args[0].Resource->K != SpirvResource::AccelerationStructure) {
      fail("TraceRayInline requires an acceleration structure, flags, mask and RayDesc", E->getExprLoc()); return {};
    }
    SpirvValue Ray = load(Args[3]);
    const SpirvType Info = Types[Ray.Type];
    if (Info.K != SpirvType::Struct || Info.Members.size() != 4) {
      fail("TraceRayInline requires a RayDesc value", E->getExprLoc()); return {};
    }
    uint32_t StaticFlags = shaderTemplateCount(Object.ASTType);
    uint32_t Flags = convert(Args[1], integerType()).Id;
    if (StaticFlags) Flags = valueOp(OpBitwiseOr, integerType(), {Flags, constant(integerType(), StaticFlags)});
    uint32_t Origin = valueOp(OpCompositeExtract, Info.Members[0], {Ray.Id, 0});
    uint32_t TMin = valueOp(OpCompositeExtract, Info.Members[1], {Ray.Id, 1});
    uint32_t Direction = valueOp(OpCompositeExtract, Info.Members[2], {Ray.Id, 2});
    uint32_t TMax = valueOp(OpCompositeExtract, Info.Members[3], {Ray.Id, 3});
    Body.emit(OpRayQueryInitializeKHR, {Query, load(Args[0]).Id, Flags,
        convert(Args[2], integerType()).Id, Origin, TMin, Direction, TMax});
    InitializedRayQueries.insert(Query);
    return {};
  }
  if (!InitializedRayQueries.contains(Query)) {
    fail("TraceRayInline must dominate every RayQuery operation", E->getExprLoc()); return {};
  }
  if (Name == "CommitProceduralPrimitiveHit") {
    if (Args.size() != 1) { fail("CommitProceduralPrimitiveHit requires a distance", E->getExprLoc()); return {}; }
    Body.emit(OpRayQueryGenerateIntersectionKHR, {Query, convert(Args[0], floatType()).Id}); return {};
  }
  if (!Args.empty()) { fail("Unexpected RayQuery method arguments", E->getExprLoc()); return {}; }
  if (Name == "Abort") { Body.emit(OpRayQueryTerminateKHR, {Query}); return {}; }
  if (Name == "CommitNonOpaqueTriangleHit") { Body.emit(OpRayQueryConfirmIntersectionKHR, {Query}); return {}; }

  uint32_t Op = StringSwitch<uint32_t>(Name)
      .Case("Proceed", OpRayQueryProceedKHR)
      .Case("RayTMin", OpRayQueryGetRayTMinKHR)
      .Case("RayFlags", OpRayQueryGetRayFlagsKHR)
      .Case("WorldRayOrigin", OpRayQueryGetWorldRayOriginKHR)
      .Case("WorldRayDirection", OpRayQueryGetWorldRayDirectionKHR).Default(0);
  std::vector<uint32_t> Operands{Query};
  bool Invert = false;
  if (!Op) {
    bool Committed = Name.starts_with("Committed");
    StringRef Property = Name.drop_front(9);
    if (Name == "CandidateType" || Name == "CommittedStatus") Op = OpRayQueryGetIntersectionTypeKHR;
    else Op = StringSwitch<uint32_t>(Property)
        .Cases({"TriangleRayT", "RayT"}, OpRayQueryGetIntersectionTKHR)
        .Case("InstanceID", OpRayQueryGetIntersectionInstanceCustomIndexKHR)
        .Case("InstanceIndex", OpRayQueryGetIntersectionInstanceIdKHR)
        .Case("GeometryIndex", OpRayQueryGetIntersectionGeometryIndexKHR)
        .Case("PrimitiveIndex", OpRayQueryGetIntersectionPrimitiveIndexKHR)
        .Case("TriangleBarycentrics", OpRayQueryGetIntersectionBarycentricsKHR)
        .Case("TriangleFrontFace", OpRayQueryGetIntersectionFrontFaceKHR)
        .Case("ProceduralPrimitiveNonOpaque", OpRayQueryGetIntersectionCandidateAABBOpaqueKHR)
        .Case("ObjectRayOrigin", OpRayQueryGetIntersectionObjectRayOriginKHR)
        .Case("ObjectRayDirection", OpRayQueryGetIntersectionObjectRayDirectionKHR)
        .Case("ObjectToWorld3x4", OpRayQueryGetIntersectionObjectToWorldKHR)
        .Case("WorldToObject3x4", OpRayQueryGetIntersectionWorldToObjectKHR).Default(0);
    if (!Op) { fail("Unsupported RayQuery method: " + Name, E->getExprLoc()); return {}; }
    Invert = Op == OpRayQueryGetIntersectionCandidateAABBOpaqueKHR;
    // CandidateAABBOpaque has no candidate/committed operand.
    if (!Invert) Operands.push_back(constant(integerType(), Committed ? 1 : 0));
  }
  uint32_t T = type(E->getType());
  uint32_t Result = valueOp(Op, T, Operands);
  if (Invert) Result = valueOp(OpLogicalNot, boolType(), {Result});
  return {Result, T, StorageClass::Function, E->getType()};
}
}
