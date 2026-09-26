//===-- ProphetSpirvEmitter.Stages.cpp - Shared entry interfaces ----------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"
#include "llvm/ADT/StringSwitch.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/TargetParser/Triple.h"

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

uint32_t ProphetSpirvEmitter::stage(const FunctionDecl *F) const {
  StringRef Name = shaderAnnotation(F, "prophet.stage:");
  if (const auto *A = F->getAttr<HLSLShaderAttr>())
    Name = Triple::getEnvironmentTypeName(A->getType());
  return StringSwitch<uint32_t>(Name)
      .Case("compute", 1).Case("vertex", 2).Cases({"pixel", "fragment"}, 3)
      .Case("geometry", 4).Case("hull", 5).Case("domain", 6)
      .Cases({"task", "amplification"}, 7).Case("mesh", 8)
      .Case("raygeneration", 10).Case("intersection", 11).Case("anyhit", 12)
      .Case("closesthit", 13).Case("miss", 14).Case("callable", 15)
      .Default(RequestedStage);
}
uint32_t ProphetSpirvEmitter::executionModel(uint32_t Stage) {
  switch (Stage) {
  case 1: return ExecutionModel::GLCompute;
  case 2: return ExecutionModel::Vertex;
  case 3: return ExecutionModel::Fragment;
  case 4: capability(Capability::Geometry); return ExecutionModel::Geometry;
  case 5: capability(Capability::Tessellation); return ExecutionModel::TessellationControl;
  case 6: capability(Capability::Tessellation); return ExecutionModel::TessellationEvaluation;
  case 7: case 8:
    capability(Capability::MeshShadingEXT); extension("SPV_EXT_mesh_shader");
    return Stage == 7 ? ExecutionModel::TaskEXT : ExecutionModel::MeshEXT;
  case 10: case 11: case 12: case 13: case 14: case 15:
    capability(Capability::RayTracingKHR); extension("SPV_KHR_ray_tracing");
    return Stage == 10 ? ExecutionModel::RayGenerationKHR :
        Stage == 11 ? ExecutionModel::IntersectionKHR : Stage == 12 ? ExecutionModel::AnyHitKHR :
        Stage == 13 ? ExecutionModel::ClosestHitKHR : Stage == 14 ? ExecutionModel::MissKHR : ExecutionModel::CallableKHR;
  default: fail("invalid shader execution stage"); return 0;
  }
}
void ProphetSpirvEmitter::executionMode(uint32_t Mode, ArrayRef<uint32_t> Args) {
  std::vector<uint32_t> Operands{CurrentEntry, Mode};
  Operands.insert(Operands.end(), Args.begin(), Args.end()); Modes.emit(OpExecutionMode, Operands);
}
std::string ProphetSpirvEmitter::semantic(const Decl *D) const {
  if (!D) return {};
  StringRef Annotation = shaderAnnotation(D, "prophet.semantic:");
  if (!Annotation.empty()) return Annotation.str();
  if (const auto *A = D->getAttr<HLSLParsedSemanticAttr>()) {
    std::string S = A->getSemanticName().str();
    if (A->getSemanticIndex()) S += std::to_string(A->getSemanticIndex());
    return S;
  }
  return {};
}
uint32_t ProphetSpirvEmitter::semanticIndex(StringRef S) const {
  size_t Start = S.size();
  while (Start && isDigit(S[Start - 1])) --Start;
  uint32_t Index = 0;
  if (Start < S.size()) S.drop_front(Start).getAsInteger(10, Index);
  return Index;
}
uint32_t ProphetSpirvEmitter::builtin(uint32_t Builtin, uint32_t T, uint32_t Storage) {
  auto Key = std::make_tuple(CurrentEntry, Builtin, Storage);
  if (auto I = Builtins.find(Key); I != Builtins.end()) {
    UsedGlobals.insert(I->second); return I->second;
  }
  uint32_t V = globalVariable(T, Storage, "builtin." + std::to_string(Builtin));
  decorate(V, Decoration::BuiltIn, {Builtin}); trackBuiltin(V, Builtin); UsedGlobals.insert(V); Builtins[Key] = V;
  return V;
}
bool ProphetSpirvEmitter::interface(const ValueDecl *D, QualType T, uint32_t Storage,
    std::vector<uint32_t> Path, uint32_t ArraySize, bool Patch, bool Primitive,
    StringRef OverrideSemantic) {
  T = T.getNonReferenceType();
  std::string Semantic = OverrideSemantic.empty() ? semantic(D) : OverrideSemantic.str();
  if (const auto *R = T->getAsRecordDecl(); R && shaderAnnotation(R, "prophet.vector:").empty()) {
    for (const FieldDecl *F : R->fields()) {
      auto FieldPath = Path; FieldPath.push_back(F->getFieldIndex());
      if (!interface(F, F->getType(), Storage, FieldPath, ArraySize, Patch, Primitive)) return false;
      // Each flattened leaf remains associated with the entry parameter/return.
      for (auto &I : Interfaces)
        if (I.Decl == F) I.Decl = D;
    }
    return true;
  }
  if (Semantic.empty()) return fail("shader interface value requires a semantic", D ? D->getLocation() : SourceLocation());
  StringRef Base(Semantic);
  while (!Base.empty() && isDigit(Base.back())) Base = Base.drop_back();
  std::string Upper = Base.upper();
  uint32_t Builtin = ~0u;
  uint32_t Type = intrinsicType(T);
  if (Upper == "SV_POSITION") Builtin = CurrentStage == 3 && Storage == StorageClass::Input ? BuiltIn::FragCoord : BuiltIn::Position;
  else if (Upper == "SV_VERTEXID") Builtin = BuiltIn::VertexIndex;
  else if (Upper == "SV_INSTANCEID") Builtin = BuiltIn::InstanceIndex;
  else if (Upper == "SV_DISPATCHTHREADID") Builtin = BuiltIn::GlobalInvocationId;
  else if (Upper == "SV_GROUPID") Builtin = BuiltIn::WorkgroupId;
  else if (Upper == "SV_GROUPTHREADID") Builtin = BuiltIn::LocalInvocationId;
  else if (Upper == "SV_GROUPINDEX") Builtin = BuiltIn::LocalInvocationIndex;
  else if (Upper == "SV_OUTPUTCONTROLPOINTID" || Upper == "SV_GSINSTANCEID") Builtin = BuiltIn::InvocationId;
  else if (Upper == "SV_DOMAINLOCATION") Builtin = BuiltIn::TessCoord;
  else if (Upper == "SV_PRIMITIVEID") Builtin = BuiltIn::PrimitiveId;
  else if (Upper == "SV_ISFRONTFACE") Builtin = BuiltIn::FrontFacing;
  else if (Upper == "SV_SAMPLEINDEX") { Builtin = BuiltIn::SampleId; capability(Capability::SampleRateShading); }
  else if (Upper == "SV_DEPTH" || Upper == "SV_DEPTHGREATEREQUAL" || Upper == "SV_DEPTHLESSEQUAL") {
    Builtin = BuiltIn::FragDepth; executionMode(ExecutionMode::DepthReplacing);
    if (Upper == "SV_DEPTHGREATEREQUAL") executionMode(ExecutionMode::DepthGreater);
    if (Upper == "SV_DEPTHLESSEQUAL") executionMode(ExecutionMode::DepthLess);
  } else if (Upper == "SV_RENDERTARGETARRAYINDEX") { Builtin = BuiltIn::Layer; capability(Capability::Geometry); }
  else if (Upper == "SV_VIEWPORTARRAYINDEX") { Builtin = BuiltIn::ViewportIndex; capability(Capability::MultiViewport); }
  else if (Upper == "SV_CLIPDISTANCE") { Builtin = BuiltIn::ClipDistance; capability(Capability::ClipDistance); }
  else if (Upper == "SV_CULLDISTANCE") { Builtin = BuiltIn::CullDistance; capability(Capability::CullDistance); }
  else if (Upper == "SV_TESSFACTOR") { Builtin = BuiltIn::TessLevelOuter; Patch = true; Type = arrayType(floatType(), 4, SpirvLayout::Natural); }
  else if (Upper == "SV_INSIDETESSFACTOR") { Builtin = BuiltIn::TessLevelInner; Patch = true; Type = arrayType(floatType(), 2, SpirvLayout::Natural); }
  else if (Upper == "SV_VIEWID") {
    Builtin = BuiltIn::ViewIndex; capability(Capability::MultiView); extension("SPV_KHR_multiview");
  } else if (Upper.starts_with("SV_") && Upper != "SV_TARGET")
    return fail("unsupported shader interface semantic: " + Semantic, D ? D->getLocation() : SourceLocation());
  if (Builtin == BuiltIn::TessCoord) Type = vectorType(floatType(), 3);
  if (Builtin == BuiltIn::VertexIndex || Builtin == BuiltIn::InstanceIndex ||
      Builtin == BuiltIn::PrimitiveId || Builtin == BuiltIn::InvocationId ||
      Builtin == BuiltIn::SampleId || Builtin == BuiltIn::Layer || Builtin == BuiltIn::ViewportIndex)
    Type = integerType(32, true);
  uint32_t ElementType = Type;
  if (ArraySize && !Patch) Type = arrayType(Type, ArraySize, SpirvLayout::Natural);
  uint32_t V = Builtin != ~0u && !ArraySize
      ? builtin(Builtin, Type, Storage)
      : globalVariable(Type, Storage, (Storage == StorageClass::Input ? "in." : "out.") + Semantic);
  if (Builtin != ~0u) Builtins[{CurrentEntry, Builtin, Storage}] = V;
  std::string Retained = StringRef(Semantic).upper();
  if (!isDigit(Retained.back()) && (!StringRef(Retained).starts_with("SV_") || Upper == "SV_TARGET")) Retained += "0";
  if (Environment == 1) extension("SPV_GOOGLE_hlsl_functionality1");
  Decorations.string(OpDecorate, {V, Decoration::UserSemantic}, Retained);
  if (Builtin != ~0u) { if (ArraySize) decorate(V, Decoration::BuiltIn, {Builtin}); }
  else {
    uint32_t Location = 0;
    if (const auto *A = D ? D->getAttr<HLSLVkLocationAttr>() : nullptr) Location = A->getLocation();
    else if (Upper == "SV_TARGET") Location = semanticIndex(Semantic);
    else Location = Storage == StorageClass::Input ? NextInput : NextOutput;
    std::function<uint32_t(uint32_t)> Slots = [&](uint32_t T) -> uint32_t {
      const auto Info = Types[T];
      if (Info.K == SpirvType::Array || Info.K == SpirvType::Matrix) return Info.Count * Slots(Info.Element);
      if (Info.K == SpirvType::Vector) return (Info.Count * Types[Info.Element].Bits + 127) / 128;
      return 1;
    };
    uint32_t &Next = Storage == StorageClass::Input ? NextInput : NextOutput;
    Next = std::max(Next, Location + Slots(ElementType));
    decorate(V, Decoration::Location, {Location});
    // Names retain semantic identity even when each stage has a different AST.
    if (Storage == StorageClass::Input && CurrentStage == 3) {
      SpirvType Scalar = Types[ElementType]; if (Scalar.K == SpirvType::Vector) Scalar = Types[Scalar.Element];
      if (Scalar.K == SpirvType::Int || Primitive || (D && hasShaderAnnotation(D, "prophet.nointerpolation")))
        decorate(V, Decoration::Flat);
    }
  }
  if (Patch) decorate(V, Decoration::Patch);
  if (Primitive) decorate(V, Decoration::PerPrimitiveEXT);
  if (D && hasShaderAnnotation(D, "prophet.noperspective")) decorate(V, Decoration::NoPerspective);
  if (D && hasShaderAnnotation(D, "prophet.centroid")) decorate(V, Decoration::Centroid);
  if (D && hasShaderAnnotation(D, "prophet.sample")) { decorate(V, Decoration::Sample); capability(Capability::SampleRateShading); }
  Interfaces.push_back({D, T, std::move(Path), V, ElementType, Storage, ArraySize, Patch, Primitive, Semantic});
  UsedGlobals.insert(V); return Diagnostic.empty();
}
SpirvValue ProphetSpirvEmitter::interfaceInput(const ValueDecl *D, QualType T, uint32_t Index) {
  SpirvValue Result = temporary(T);
  for (const auto &I : Interfaces) {
    if (I.Decl != D || I.Storage != StorageClass::Input) continue;
    SpirvValue Input{I.Variable, I.Type, StorageClass::Input, I.ASTType, true};
    if (I.ArraySize && !I.Patch) {
      Input = {valueOp(OpAccessChain, pointerType(I.Type, StorageClass::Input),
          {I.Variable, Index ? Index : constant(integerType(), 0)}), I.Type, StorageClass::Input, I.ASTType, true};
    }
    SpirvValue Destination = Result;
    if (!I.Path.empty()) {
      std::vector<uint32_t> Path;
      for (uint32_t P : I.Path) Path.push_back(constant(integerType(), P));
      Destination = access(Result, Path, I.ASTType);
    }
    if (Types[Input.Type].K == SpirvType::Array && Input.Type != Destination.Type) {
      if (Types[Destination.Type].K == SpirvType::Float)
        store(Destination, access(Input, {constant(integerType(), 0)}, Ctx.FloatTy));
      else for (uint32_t J = 0; J < Types[Destination.Type].Count; ++J)
        store(access(Destination, {constant(integerType(), J)}, Ctx.FloatTy),
              access(Input, {constant(integerType(), J)}, Ctx.FloatTy));
    } else store(Destination, Input);
  }
  return Result;
}
bool ProphetSpirvEmitter::interfaceOutput(const ValueDecl *D, SpirvValue V, uint32_t Index) {
  for (const auto &I : Interfaces) {
    if (I.Decl != D || I.Storage != StorageClass::Output) continue;
    SpirvValue Source = V;
    if (!I.Path.empty()) {
      if (V.Address) {
        std::vector<uint32_t> Path;
        for (uint32_t P : I.Path) Path.push_back(constant(integerType(), P));
        Source = access(V, Path, I.ASTType);
      } else Source = extract(V, I.Path, I.ASTType);
    }
    SpirvValue Output{I.Variable, I.Type, StorageClass::Output, I.ASTType, true};
    if (I.ArraySize && !I.Patch)
      Output = {valueOp(OpAccessChain, pointerType(I.Type, StorageClass::Output),
          {I.Variable, Index ? Index : constant(integerType(), 0)}), I.Type, StorageClass::Output, I.ASTType, true};
    if (Types[Output.Type].K == SpirvType::Array && Source.Type != Output.Type) {
      store(Output, {zero(Output.Type), Output.Type});
      if (Types[Source.Type].K == SpirvType::Float)
        store(access(Output, {constant(integerType(), 0)}, Ctx.FloatTy), Source);
      else for (uint32_t J = 0; J < Types[Source.Type].Count; ++J)
        store(access(Output, {constant(integerType(), J)}, Ctx.FloatTy),
              Source.Address ? access(Source, {constant(integerType(), J)}, Ctx.FloatTy)
                             : extract(Source, {J}, Ctx.FloatTy));
    } else if (!store(Output, Source)) return false;
  }
  return Diagnostic.empty();
}
bool ProphetSpirvEmitter::regularEntry(const FunctionDecl *F) {
  std::vector<SpirvValue> Args;
  std::vector<std::pair<const ParmVarDecl *, SpirvValue>> Outputs;
  for (const ParmVarDecl *P : F->parameters()) {
    const auto *Direction = P->getAttr<HLSLParamModifierAttr>();
    StringRef CppDirection = shaderAnnotation(P, "prophet.direction:");
    bool Output = (Direction && (Direction->isOut() || Direction->isInOut())) || CppDirection == "out" || CppDirection == "inout";
    bool Input = CppDirection != "out" && (!Direction || !Direction->isOut() || Direction->isInOut());
    if (Input && !interface(P, P->getType(), StorageClass::Input)) return false;
    if (Output && !interface(P, P->getType(), StorageClass::Output)) return false;
    SpirvValue Argument = Input ? interfaceInput(P, P->getType().getNonReferenceType()) : temporary(P->getType());
    Args.push_back(Argument);
    if (Output) Outputs.emplace_back(P, Argument);
  }
  if (!F->getReturnType()->isVoidType() && !interface(F, F->getReturnType(), StorageClass::Output)) return false;
  SpirvValue Result = callFunction(F, Args);
  if (!F->getReturnType()->isVoidType() && !interfaceOutput(F, Result)) return false;
  for (auto [P, V] : Outputs) if (!interfaceOutput(P, V)) return false;
  return Diagnostic.empty();
}
bool ProphetSpirvEmitter::entry(const FunctionDecl *F) {
  CurrentFunction = F; CurrentStage = stage(F); CurrentEntry = id();
  RaySlots.clear(); MeshPayload = {};
  if (RequestedStage == 9 && CurrentStage < 10)
    return fail("shader library entries must be ray tracing stages", F->getLocation());
  WorkgroupInvocations = 1; ControlConditions.clear(); UniformLoops.clear();
  if (RequestedStage != 9 && RequestedStage != CurrentStage)
    return fail("shader stage does not match the requested entry", F->getLocation());
  uint32_t Model = executionModel(CurrentStage), Start = id();
  CurrentLabel = Start; Terminated = false;
  Body.Words.clear(); LocalVariables.Words.clear(); Locals.clear(); Calls.clear();
  Interfaces.clear(); UsedGlobals.clear(); FlowTargets.clear(); NextInput = NextOutput = 0;
  if (CurrentStage == 1 || CurrentStage == 7 || CurrentStage == 8) {
    uint32_t X = 1, Y = 1, Z = 1;
    if (const auto *A = F->getAttr<HLSLNumThreadsAttr>()) { X = A->getX(); Y = A->getY(); Z = A->getZ(); }
    else {
      StringRef ThreadAnnotation = shaderAnnotation(F, "prophet.numthreads:");
      SmallVector<StringRef, 3> Parts; ThreadAnnotation.split(Parts, ',');
      if (Parts.size() != 3 || Parts[0].getAsInteger(10, X) || Parts[1].getAsInteger(10, Y) || Parts[2].getAsInteger(10, Z))
        return fail("compute/task/mesh entry requires numthreads", F->getLocation());
    }
    if (!X || !Y || !Z || uint64_t(X) * Y * Z > 1024)
      return fail("invalid shader workgroup dimensions", F->getLocation());
    executionMode(ExecutionMode::LocalSize, {X, Y, Z});
    WorkgroupInvocations = X * Y * Z;
  }
  if (CurrentStage == 3) executionMode(ExecutionMode::OriginUpperLeft);
  for (const VarDecl *D : GlobalSemanticDecls) {
    if (hasShaderAnnotation(D, "prophet.global_invocation_id_x")) {
      uint32_t T = vectorType(integerType(), 3);
      uint32_t V = builtin(BuiltIn::GlobalInvocationId, T);
      SpirvValue Bound{V, T, StorageClass::Input, D->getType(), true}; Bound.Swizzle = {0};
      Globals[D->getCanonicalDecl()] = Bound;
    } else {
      std::string S = semantic(D);
      bool Out = CurrentStage == 2 ? !StringRef(S).starts_with_insensitive("SV_VertexID") &&
          !StringRef(S).starts_with_insensitive("SV_InstanceID") :
          StringRef(S).starts_with_insensitive("SV_Target") || StringRef(S).starts_with_insensitive("SV_Depth");
      uint32_t Storage = Out ? StorageClass::Output : StorageClass::Input;
      if (!interface(D, D->getType(), Storage)) return false;
      Globals[D->getCanonicalDecl()] = interfaceReference(D, D->getType(), Storage);
    }
  }
  bool OK = CurrentStage == 4 ? geometryEntry(F) : CurrentStage == 5 ? hullEntry(F) :
      CurrentStage == 6 ? domainEntry(F) : CurrentStage == 8 ? meshEntry(F) :
      CurrentStage >= 10 ? rayEntry(F) : regularEntry(F);
  if (!OK || !Diagnostic.empty()) return false;
  if (!Terminated) Body.emit(OpReturn);
  uint32_t Void = voidType();
  uint32_t FunctionType = internType(OpTypeFunction, {Void}, {});
  Functions.emit(OpFunction, {Void, CurrentEntry, 0, FunctionType});
  Functions.emit(OpLabel, {Start}); Functions.append(LocalVariables); Functions.append(Body);
  Functions.emit(OpFunctionEnd); name(CurrentEntry, F->getName());
  std::vector<uint32_t> Variables;
  for (uint32_t V : UsedGlobals) {
    if (Environment != 1) Variables.push_back(V);
    else {
      bool IO = false;
      for (const auto &I : Interfaces) IO |= I.Variable == V;
      for (const auto &[K, B] : Builtins) IO |= B == V;
      if (IO) Variables.push_back(V);
    }
  }
  Entries.string(OpEntryPoint, {Model, CurrentEntry}, F->getName(), Variables);
  return true;
}
}
