//===-- ProphetSpirvEmitter.AdvancedStages.cpp - AST graphics/ray stages ---===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"
#include "llvm/ADT/StringSwitch.h"

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

static StringRef domain(const FunctionDecl *F) {
  if (const auto *A = F->getAttr<HLSLTessDomainAttr>()) return A->getDomain();
  return shaderAnnotation(F, "prophet.domain:");
}
static StringRef topology(const FunctionDecl *F) {
  if (const auto *A = F->getAttr<HLSLTessOutputTopologyAttr>()) return A->getTopology();
  return shaderAnnotation(F, "prophet.outputtopology:");
}
static QualType arrayAST(ASTContext &Ctx, QualType Element, uint32_t Count) {
  return Ctx.getConstantArrayType(Element, APInt(64, Count), nullptr, ArraySizeModifier::Normal, 0);
}
static StringRef patchKind(QualType T) {
  if (const auto *P = shaderTemplate(T)) return P->getName();
  return {};
}
bool ProphetSpirvEmitter::geometryEntry(const FunctionDecl *F) {
  uint32_t MaxVertices = 0;
  if (const auto *A = F->getAttr<HLSLMaxVertexCountAttr>()) MaxVertices = A->getCount();
  else shaderAnnotation(F, "prophet.maxvertexcount:").getAsInteger(10, MaxVertices);
  if (!MaxVertices) return fail("geometry entry requires a maxvertexcount attribute", F->getLocation());
  executionMode(ExecutionMode::OutputVertices, {MaxVertices});
  executionMode(ExecutionMode::Invocations, {1});
  std::vector<SpirvValue> Args;
  bool InputSeen = false, StreamSeen = false;
  for (const ParmVarDecl *P : F->parameters()) {
    StringRef Primitive = shaderAnnotation(P, "prophet.primitive:");
    if (const auto *A = P->getAttr<HLSLGeometryInputAttr>()) Primitive = A->getSpelling();
    const auto *Array = Ctx.getAsConstantArrayType(P->getOriginalType().getNonReferenceType());
    StringRef Class = patchKind(P->getType());
    if (!Primitive.empty() && Array) {
      uint32_t Count = StringSwitch<uint32_t>(Primitive).Case("point", 1).Case("line", 2)
          .Case("triangle", 3).Case("lineadj", 4).Case("triangleadj", 6).Default(0);
      uint32_t Mode = StringSwitch<uint32_t>(Primitive).Case("point", ExecutionMode::InputPoints)
          .Case("line", ExecutionMode::InputLines).Case("triangle", ExecutionMode::Triangles)
          .Case("lineadj", ExecutionMode::InputLinesAdjacency).Case("triangleadj", ExecutionMode::InputTrianglesAdjacency).Default(0);
      if (!Count || Array->getSize().getLimitedValue() != Count)
        return fail((Primitive + " input must be an array of exactly " + Twine(Count) + " records").str(), P->getLocation());
      executionMode(Mode); InputSeen = true;
      if (!interface(P, Array->getElementType(), StorageClass::Input, {}, Count)) return false;
      Args.push_back(interfaceReference(P, P->getOriginalType().getNonReferenceType(), StorageClass::Input));
    } else if (Class.ends_with("Stream")) {
      if (StreamSeen) return fail("geometry entry currently requires one output stream", P->getLocation());
      uint32_t Mode = Class == "PointStream" ? ExecutionMode::OutputPoints :
          Class == "LineStream" ? ExecutionMode::OutputLineStrip : ExecutionMode::OutputTriangleStrip;
      executionMode(Mode); StreamSeen = true;
      if (!interface(nullptr, shaderTemplateElement(P->getType()), StorageClass::Output)) return false;
      SpirvValue Stream; Stream.Stream = true; Stream.ASTType = P->getType(); Args.push_back(Stream);
    } else {
      if (!interface(P, P->getType(), StorageClass::Input)) return false;
      Args.push_back(interfaceInput(P, P->getType()));
    }
  }
  if (!InputSeen || !StreamSeen) return fail("geometry entry requires primitive input and output stream", F->getLocation());
  callFunction(F, Args); return Diagnostic.empty();
}
bool ProphetSpirvEmitter::hullEntry(const FunctionDecl *F) {
  uint32_t Count = 0;
  if (const auto *A = F->getAttr<HLSLTessOutputControlPointsAttr>()) Count = A->getCount();
  else shaderAnnotation(F, "prophet.outputcontrolpoints:").getAsInteger(10, Count);
  if (!Count || Count > 32) return fail("hull outputcontrolpoints must be in 1..32", F->getLocation());
  executionMode(ExecutionMode::OutputVertices, {Count});
  StringRef Domain = domain(F), Partitioning = shaderAnnotation(F, "prophet.partitioning:");
  if (const auto *A = F->getAttr<HLSLTessPartitioningAttr>()) Partitioning = A->getPartitioning();
  if (Domain.empty() || Partitioning.empty()) return fail("hull entry requires domain and partitioning", F->getLocation());
  executionMode(Domain == "quad" ? ExecutionMode::Quads : Domain == "isoline" ? ExecutionMode::Isolines : ExecutionMode::Triangles);
  executionMode(Partitioning == "fractional_even" ? ExecutionMode::SpacingFractionalEven :
      Partitioning == "fractional_odd" ? ExecutionMode::SpacingFractionalOdd : ExecutionMode::SpacingEqual);
  executionMode(topology(F) == "triangle_cw" ? ExecutionMode::VertexOrderCw : ExecutionMode::VertexOrderCcw);

  SpirvValue InputPatch;
  uint32_t Control = isolatedFunction(F->getNameAsString() + ".control_points", [&] {
    std::vector<SpirvValue> Args;
    for (const ParmVarDecl *P : F->parameters()) {
      if (patchKind(P->getType()) == "InputPatch") {
        QualType Element = shaderTemplateElement(P->getType()); uint32_t N = shaderTemplateCount(P->getType());
        if (!N || N > 32) return fail("invalid hull input patch size", P->getLocation());
        if (!interface(P, Element, StorageClass::Input, {}, N)) return false;
        InputPatch = interfaceReference(P, arrayAST(Ctx, Element, N), StorageClass::Input);
        Args.push_back(InputPatch);
      } else {
        if (!interface(P, P->getType(), StorageClass::Input)) return false;
        Args.push_back(interfaceInput(P, P->getType()));
      }
    }
    if (!interface(F, F->getReturnType(), StorageClass::Output, {}, Count)) return false;
    uint32_t Invocation = load({builtin(BuiltIn::InvocationId, integerType(32, true)),
        integerType(32, true), StorageClass::Input, Ctx.IntTy, true}).Id;
    return interfaceOutput(F, callFunction(F, Args), Invocation);
  });
  if (!Control) return false;

  StringRef PatchName = shaderAnnotation(F, "prophet.patchconstantfunc:");
  if (const auto *A = F->getAttr<HLSLTessPatchConstantFuncAttr>()) PatchName = A->getFunction();
  const FunctionDecl *PCF = nullptr;
  for (const Decl *D : F->getDeclContext()->decls())
    if (const auto *Candidate = dyn_cast<FunctionDecl>(D))
      if (Candidate->getNameAsString() == PatchName && Candidate->doesThisDeclarationHaveABody()) PCF = Candidate;
  if (!PCF) return fail("hull patch constant function was not found", F->getLocation());
  if (!interface(PCF, PCF->getReturnType(), StorageClass::Output, {}, 0, true)) return false;
  uint32_t Patch = isolatedFunction(PCF->getNameAsString() + ".patch_constants", [&] {
    std::vector<SpirvValue> Args;
    for (const ParmVarDecl *P : PCF->parameters()) {
      if (patchKind(P->getType()) == "InputPatch") {
        if (!InputPatch) return fail("patch function input is unavailable", P->getLocation());
        Args.push_back(InputPatch);
      } else if (patchKind(P->getType()) == "OutputPatch")
        Args.push_back(interfaceReference(F, arrayAST(Ctx, F->getReturnType(), Count), StorageClass::Output));
      else {
        if (!interface(P, P->getType(), StorageClass::Input)) return false;
        Args.push_back(interfaceInput(P, P->getType()));
      }
    }
    return interfaceOutput(PCF, callFunction(PCF, Args));
  });
  if (!Patch) return false;
  valueOp(OpFunctionCall, voidType(), {Control});
  Body.emit(OpControlBarrier, {constant(integerType(), Scope::Workgroup),
      constant(integerType(), Scope::Invocation), constant(integerType(), 0)});
  uint32_t Invocation = load({builtin(BuiltIn::InvocationId, integerType(32, true)),
      integerType(32, true), StorageClass::Input, Ctx.IntTy, true}).Id;
  uint32_t First = valueOp(OpIEqual, boolType(), {Invocation, constant(integerType(32, true), 0)});
  uint32_t PatchBlock = id(), Merge = id();
  Body.emit(OpSelectionMerge, {Merge, 0}); conditionalBranch(First, PatchBlock, Merge); label(PatchBlock);
  valueOp(OpFunctionCall, voidType(), {Patch}); branch(Merge); label(Merge);
  return Diagnostic.empty();
}
bool ProphetSpirvEmitter::domainEntry(const FunctionDecl *F) {
  StringRef Domain = domain(F);
  if (Domain.empty()) return fail("domain entry requires a domain attribute", F->getLocation());
  executionMode(Domain == "quad" ? ExecutionMode::Quads : Domain == "isoline" ? ExecutionMode::Isolines : ExecutionMode::Triangles);
  executionMode(ExecutionMode::SpacingEqual); executionMode(ExecutionMode::VertexOrderCcw);
  std::vector<SpirvValue> Args;
  for (const ParmVarDecl *P : F->parameters()) {
    if (patchKind(P->getType()) == "OutputPatch") {
      QualType Element = shaderTemplateElement(P->getType()); uint32_t N = shaderTemplateCount(P->getType());
      if (!N || N > 32) return fail("invalid domain patch size", P->getLocation());
      if (!interface(P, Element, StorageClass::Input, {}, N)) return false;
      Args.push_back(interfaceReference(P, arrayAST(Ctx, Element, N), StorageClass::Input));
    } else {
      bool Patch = P->getType().getNonReferenceType()->isRecordType() && semantic(P).empty();
      if (!interface(P, P->getType(), StorageClass::Input, {}, 0, Patch)) return false;
      Args.push_back(interfaceInput(P, P->getType()));
    }
  }
  if (!interface(F, F->getReturnType(), StorageClass::Output)) return false;
  return interfaceOutput(F, callFunction(F, Args));
}
bool ProphetSpirvEmitter::meshEntry(const FunctionDecl *F) {
  StringRef Topology = topology(F);
  uint32_t OutputMode = Topology == "point" ? ExecutionMode::OutputPoints :
      Topology == "line" ? ExecutionMode::OutputLinesEXT : ExecutionMode::OutputTrianglesEXT;
  if (Topology != "point" && Topology != "line" && Topology != "triangle")
    return fail("mesh entry requires point, line or triangle outputtopology", F->getLocation());
  executionMode(OutputMode);
  std::vector<SpirvValue> Args;
  bool HasVertices = false, HasIndices = false;
  for (const ParmVarDecl *P : F->parameters()) {
    StringRef Kind = shaderAnnotation(P, "prophet.mesh:");
    if (const auto *A = P->getAttr<HLSLMeshParamModifierAttr>()) Kind = A->getSpelling();
    if (Kind == "payload") {
      QualType T = P->getType().getNonReferenceType(); uint32_t Type = type(T);
      uint32_t V = globalVariable(Type, StorageClass::TaskPayloadWorkgroupEXT, "task.payload");
      UsedGlobals.insert(V); Args.push_back({V, Type, StorageClass::TaskPayloadWorkgroupEXT, T, true});
      continue;
    }
    if (Kind == "vertices" || Kind == "indices" || Kind == "primitives") {
      const auto *A = Ctx.getAsConstantArrayType(P->getOriginalType().getNonReferenceType());
      if (!A || !A->getSize().getLimitedValue() || A->getSize().getLimitedValue() > 256)
        return fail("mesh output must be a nonempty constant array of at most 256 entries", P->getLocation());
      uint32_t N = uint32_t(A->getSize().getLimitedValue()); QualType Element = A->getElementType();
      if (Kind == "indices") {
        uint32_t T = intrinsicType(Element), VariableType = arrayType(T, N, SpirvLayout::Natural);
        uint32_t BI = Topology == "point" ? BuiltIn::PrimitivePointIndicesEXT :
            Topology == "line" ? BuiltIn::PrimitiveLineIndicesEXT : BuiltIn::PrimitiveTriangleIndicesEXT;
        uint32_t V = globalVariable(VariableType, StorageClass::Output, "mesh.indices");
        decorate(V, Decoration::BuiltIn, {BI}); UsedGlobals.insert(V);
        Interfaces.push_back({P, Element, {}, V, T, StorageClass::Output, N, false, false, "indices"});
        executionMode(ExecutionMode::OutputPrimitivesEXT, {N}); HasIndices = true;
      } else {
        if (!interface(P, Element, StorageClass::Output, {}, N, false, Kind == "primitives")) return false;
        if (Kind == "vertices") { executionMode(ExecutionMode::OutputVertices, {N}); HasVertices = true; }
      }
      Args.push_back(interfaceReference(P, P->getOriginalType().getNonReferenceType(), StorageClass::Output));
    } else {
      if (!interface(P, P->getType(), StorageClass::Input)) return false;
      Args.push_back(interfaceInput(P, P->getType()));
    }
  }
  if (!HasVertices || !HasIndices) return fail("mesh entry requires vertices and indices output", F->getLocation());
  // Output arguments are views over stage variables. Each invocation writes
  // only the subobjects addressed by its source statements; no array copyout.
  callFunction(F, Args); return Diagnostic.empty();
}
SpirvValue ProphetSpirvEmitter::raySlot(SpirvValue Value, uint32_t Storage, StringRef Name) {
  auto Key = std::make_pair(Storage, Value.Type);
  if (auto Found = RaySlots.find(Key); Found != RaySlots.end()) return Found->second;
  uint32_t Location = 0;
  for (const auto &[Existing, Slot] : RaySlots) if (Existing.first == Storage) ++Location;
  if (Storage == StorageClass::HitAttributeKHR && Location) {
    fail("one intersection entry must use one hit attribute type"); return {};
  }
  uint32_t Variable = globalVariable(Value.Type, Storage, Name);
  if (Storage != StorageClass::HitAttributeKHR) decorate(Variable, Decoration::Location, {Location});
  UsedGlobals.insert(Variable);
  return RaySlots[Key] = {Variable, Value.Type, Storage, Value.ASTType, true};
}
bool ProphetSpirvEmitter::rayEntry(const FunctionDecl *F) {
  if (!F->getReturnType()->isVoidType()) return fail("ray entry must return void", F->getLocation());
  if (CurrentStage == 10) {
    if (F->getNumParams()) return fail("raygeneration entry takes no parameters", F->getLocation());
    callFunction(F, {}); return Diagnostic.empty();
  }
  std::vector<SpirvValue> Args;
  for (unsigned I = 0; I < F->getNumParams(); ++I) {
    const ParmVarDecl *P = F->getParamDecl(I); QualType T = P->getType().getNonReferenceType();
    if (!T->isRecordType()) return fail("ray payload and hit attributes must be records", P->getLocation());
    if (I == 0) {
      const auto *Direction = P->getAttr<HLSLParamModifierAttr>();
      bool Mutable = Direction ? Direction->isInOut() :
          Ctx.getLangOpts().HLSLCPlusPlus &&
          P->getType()->isReferenceType() && !T.isConstQualified();
      if (!Mutable) return fail("ray payload requires inout or a mutable C++ reference", P->getLocation());
    }
    uint32_t Storage = CurrentStage == 15 ? StorageClass::IncomingCallableDataKHR :
        I == 0 ? StorageClass::IncomingRayPayloadKHR : StorageClass::HitAttributeKHR;
    uint32_t Type = type(T), V = globalVariable(Type, Storage, P->getName());
    if (Storage != StorageClass::HitAttributeKHR) decorate(V, Decoration::Location, {0});
    UsedGlobals.insert(V); Args.push_back({V, Type, Storage, T, true});
  }
  if ((CurrentStage == 14 || CurrentStage == 15) && Args.size() != 1)
    return fail("miss/callable entry requires one payload record", F->getLocation());
  if ((CurrentStage == 12 || CurrentStage == 13) && Args.size() != 2)
    return fail("hit entry requires payload and hit attribute records", F->getLocation());
  if (CurrentStage == 11 && !Args.empty()) return fail("intersection entry takes no parameters", F->getLocation());
  callFunction(F, Args); return Diagnostic.empty();
}
}
