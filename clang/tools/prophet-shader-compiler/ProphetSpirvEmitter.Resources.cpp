//===-- ProphetSpirvEmitter.Resources.cpp - AST resource bindings ---------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"
#include "prophet_shader_compiler.h"
#include "clang/AST/HLSLResource.h"
#include "clang/AST/RecordLayout.h"
#include "clang/Basic/SourceManager.h"
#include "llvm/ADT/StringExtras.h"

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

bool ProphetSpirvEmitter::binding(const NamedDecl *D, uint32_t &Set, uint32_t &Slot) {
  StringRef A = shaderAnnotation(D, "prophet.storage_buffer:");
  if (A.empty()) A = shaderAnnotation(D, "prophet.uniform_buffer:");
  if (A.empty()) A = shaderAnnotation(D, "prophet.sampled_texture:");
  if (A.empty()) A = shaderAnnotation(D, "prophet.storage_texture:");
  if (A.empty()) A = shaderAnnotation(D, "prophet.sampler:");
  if (A.empty()) A = shaderAnnotation(D, "prophet.acceleration_structure:");
  if (!A.empty()) {
    auto Parts = A.split(':');
    if (Parts.first.getAsInteger(10, Set) || Parts.second.getAsInteger(10, Slot))
      return fail("resource annotation requires set:binding", D->getLocation());
  } else {
    clang::hlsl::ResourceBindingAttrs B(D);
    if (B.isExplicit()) {
      Set = B.getSpace(); Slot = B.getSlot();
      if (!B.VkBinding && (CompileFlags & PROPHET_SHADER_SEPARATE_REGISTER_CLASSES)) {
        uint32_t Class;
        switch (B.RegBinding->getRegisterType()) {
        case HLSLResourceBindingAttr::RegisterType::UAV: Class = 0; break;
        case HLSLResourceBindingAttr::RegisterType::SRV: Class = 1; break;
        case HLSLResourceBindingAttr::RegisterType::CBuffer: Class = 2; break;
        case HLSLResourceBindingAttr::RegisterType::Sampler: Class = 3; break;
        default: return fail("shader descriptors require register t, u, b or s", D->getLocation());
        }
        if (Slot > (UINT32_MAX - Class) / 4)
          return fail("register index overflows the SPIR-V binding mapping", D->getLocation());
        Slot = 4 * Slot + Class;
      }
    } else {
      Set = B.getSpace(); Slot = 0;
      while (Bindings.contains({Set, Slot})) ++Slot;
    }
  }
  auto [It, Added] = Bindings.emplace(std::make_pair(Set, Slot), D);
  if (!Added && It->second->getCanonicalDecl() != D->getCanonicalDecl())
    return fail(("SPIR-V descriptor set " + Twine(Set) + " binding " + Twine(Slot) +
       " is assigned to both " + cast<NamedDecl>(It->second)->getName() + " and " + D->getName() + "; use [[vk::binding(binding, set)]] or unique shader bindings").str(), D->getLocation());
  return true;
}
bool ProphetSpirvEmitter::resource(const VarDecl *D) {
  QualType T = D->getType().getNonReferenceType();
  uint32_t Count = 1;
  if (const auto *A = Ctx.getAsConstantArrayType(T)) {
    Count = uint32_t(A->getSize().getLimitedValue()); T = A->getElementType();
  }
  const auto *Record = T->getAsCXXRecordDecl();
  bool BuiltinRecord = Record && (Record->isHLSLBuiltinRecord() ||
      Ctx.getSourceManager().isInSystemHeader(Record->getLocation()));
  // A user record named MyTexture or SamplerState is still an ordinary record.
  std::string Name = BuiltinRecord ? Record->getNameAsString() : "";
  bool Storage = !shaderAnnotation(D, "prophet.storage_buffer:").empty();
  bool Uniform = !shaderAnnotation(D, "prophet.uniform_buffer:").empty() || Name == "ConstantBuffer";
  bool IsBuffer = Storage || Uniform || Name == "StructuredBuffer" ||
      Name == "RWStructuredBuffer" || Name == "ByteAddressBuffer" || Name == "RWByteAddressBuffer";
  bool IsTexture = StringRef(Name).contains("Texture");
  bool IsSampler = Name == "SamplerState" || Name == "SamplerComparisonState";
  bool IsAcceleration = Name == "RaytracingAccelerationStructure" || Name == "RayTracingAccelerationStructure";
  if (!IsBuffer && !IsTexture && !IsSampler && !IsAcceleration) return false;
  auto Owned = std::make_unique<SpirvResource>();
  auto &R = *Owned;
  R.Decl = D; R.Count = Count; R.Uniform = Uniform;
  R.ElementType = shaderTemplateElement(T);
  if (R.ElementType.isNull()) R.ElementType = Ctx.UnsignedIntTy;
  if (needsDestruction(R.ElementType)) {
    fail("shader resource elements must have trivial destruction", D->getLocation()); return true;
  }
  if (!binding(D, R.Set, R.Binding)) return true;
  uint32_t StorageClass = spv::StorageClass::UniformConstant;
  uint32_t VariableType = 0;
  if (IsBuffer) {
    R.K = SpirvResource::Buffer;
    R.ReadOnly = Uniform || Name == "StructuredBuffer" || Name == "ByteAddressBuffer";
    SpirvLayout Layout = Uniform ? SpirvLayout::Uniform : SpirvLayout::Storage;
    StorageClass = Uniform ? spv::StorageClass::Uniform : spv::StorageClass::StorageBuffer;
    R.CppRecord = Storage && Record && !Record->isHLSLBuiltinRecord() &&
        (!Ctx.getLangOpts().HLSL || Ctx.getLangOpts().HLSLCPlusPlus) &&
        Name != "StructuredBuffer" && Name != "ByteAddressBuffer" && Name != "RWByteAddressBuffer";
    std::vector<uint32_t> Members;
    if (R.CppRecord) {
      if (!Record->isTriviallyCopyable() || Record->getNumBases()) {
        fail("storage_buffer must have a trivial POD layout", D->getLocation()); return true;
      }
      // Only the legacy annotated POD interface promises C++ host offsets.
      // Native StructuredBuffer<T> uses shader layout in both language modes.
      std::function<bool(QualType)> ValidateHostLayout = [&](QualType Member) {
        if (const auto *A = Ctx.getAsConstantArrayType(Member))
          return ValidateHostLayout(A->getElementType());
        const auto *Nested = Member->getAsCXXRecordDecl();
        if (!Nested) return true;
        if (Nested->getNumBases()) return fail("C++ POD buffer members cannot contain base subobjects", Nested->getLocation());
        uint32_t PhysicalType = type(Member, Layout);
        if (!Diagnostic.empty()) return false;
        const auto &Host = Ctx.getASTRecordLayout(Nested);
        for (const FieldDecl *F : Nested->fields()) {
          if (Host.getFieldOffset(F->getFieldIndex()) / 8 != Types[PhysicalType].Offsets[F->getFieldIndex()])
            return fail("C++ storage field offset differs from Vulkan layout; add explicit padding/alignment", F->getLocation());
          if (!ValidateHostLayout(F->getType())) return false;
        }
        return true;
      };
      for (const FieldDecl *F : Record->fields()) {
        if (!ValidateHostLayout(F->getType())) return true;
        if (F->getType()->isReferenceType()) {
          fail("stored reference in a GPU resource is forbidden", F->getLocation()); return true;
        }
        if (const auto *A = Ctx.getAsConstantArrayType(F->getType())) {
          R.ElementType = A->getElementType(); R.Element = type(R.ElementType, Layout);
          if (Ctx.getTypeSizeInChars(R.ElementType).getQuantity() !=
              ((Types[R.Element].Size + Types[R.Element].Align - 1) / Types[R.Element].Align) * Types[R.Element].Align) {
            fail("C++ storage element size does not match its Vulkan array stride; use an explicitly padded POD", F->getLocation()); return true;
          }
          SpirvType ElementLayout = physicalType(R.ElementType, Layout);
          Members.push_back(arrayType(R.Element, 0, Layout, true, &ElementLayout));
        } else Members.push_back(type(F->getType(), Layout));
      }
      if (Members.empty() || Types[Members.back()].K != SpirvType::RuntimeArray) {
        fail("storage_buffer requires a final data array", D->getLocation()); return true;
      }
    } else if (Uniform) {
      if (!shaderAnnotation(D, "prophet.uniform_buffer:").empty() && Name != "ConstantBuffer") R.ElementType = T;
      R.Element = type(R.ElementType, Layout); Members.push_back(R.Element);
    } else {
      R.Element = type(R.ElementType, Layout);
      SpirvType ElementLayout = physicalType(R.ElementType, Layout);
          Members.push_back(arrayType(R.Element, 0, Layout, true, &ElementLayout));
    }
    R.Type = structType(Members, Layout, true);
    if (R.CppRecord) {
      const auto &Host = Ctx.getASTRecordLayout(Record);
      for (const FieldDecl *F : Record->fields())
        if (Host.getFieldOffset(F->getFieldIndex()) / 8 != Types[R.Type].Offsets[F->getFieldIndex()]) {
          fail("C++ storage field offset differs from Vulkan layout; add explicit padding/alignment", F->getLocation()); return true;
        }
    }
    VariableType = R.Type;
  } else if (IsSampler) {
    R.K = Name == "SamplerComparisonState" ? SpirvResource::ComparisonSampler : SpirvResource::Sampler;
    SpirvType Info; Info.K = SpirvType::Sampler;
    R.Type = VariableType = internType(OpTypeSampler, {}, Info);
  } else if (IsAcceleration) {
    R.K = SpirvResource::AccelerationStructure;
    capability(Capability::RayTracingKHR); extension("SPV_KHR_ray_tracing");
    SpirvType Info; Info.K = SpirvType::Acceleration;
    R.Type = VariableType = internType(OpTypeAccelerationStructureKHR, {}, Info);
  } else {
    R.K = StringRef(Name).starts_with("RW") ? SpirvResource::StorageTexture : SpirvResource::Texture;
    R.ReadOnly = R.K == SpirvResource::Texture;
    R.Arrayed = StringRef(Name).contains("Array");
    R.Multisample = StringRef(Name).contains("MS");
    R.Dimension = StringRef(Name).contains("1D") ? Dim::_1D :
                  StringRef(Name).contains("3D") ? Dim::_3D :
                  StringRef(Name).contains("Cube") ? Dim::Cube : Dim::_2D;
    if (R.Dimension == Dim::_1D) capability(R.ReadOnly ? Capability::Sampled1D : Capability::Image1D);
    if (R.Dimension == Dim::Cube && R.Arrayed) capability(Capability::SampledCubeArray);
    R.Depth = R.ElementType->isSpecificBuiltinType(BuiltinType::Float) && R.ReadOnly;
    QualType Scalar = R.ElementType;
    if (const auto *V = Scalar->getAs<VectorType>()) Scalar = V->getElementType();
    R.Element = type(Scalar);
    uint32_t Width = 1;
    if (const auto *V = R.ElementType->getAs<VectorType>()) Width = V->getNumElements();
    uint32_t Format = ImageFormat::Unknown;
    if (!R.ReadOnly) {
      if (Width == 2) capability(Capability::StorageImageExtendedFormats);
      Format = Scalar->isRealFloatingType()
          ? (Width == 1 ? ImageFormat::R32f : Width == 2 ? ImageFormat::Rg32f : ImageFormat::Rgba32f)
          : Scalar->isSignedIntegerType()
          ? (Width == 1 ? ImageFormat::R32i : Width == 2 ? ImageFormat::Rg32i : ImageFormat::Rgba32i)
          : (Width == 1 ? ImageFormat::R32ui : Width == 2 ? ImageFormat::Rg32ui : ImageFormat::Rgba32ui);
    }
    SpirvType Info; Info.K = SpirvType::Image; Info.Element = R.Element;
    R.Type = VariableType = internType(OpTypeImage,
        {R.Element, R.Dimension, uint32_t(R.Depth), uint32_t(R.Arrayed),
         uint32_t(R.Multisample), R.ReadOnly ? 1u : 2u, Format}, Info);
  }
  if (!Diagnostic.empty()) return true;
  if (Count > 1) VariableType = arrayType(VariableType, Count, SpirvLayout::Natural);
  R.Variable = globalVariable(VariableType, StorageClass, D->getName());
  if (ObjectMode) ObjectResourceOwners[R.Variable] = D;
  if (R.K == SpirvResource::Buffer && R.ReadOnly) decorate(R.Variable, Decoration::NonWritable);
  if (D->hasAttr<HLSLGloballyCoherentAttr>()) {
    if (R.ReadOnly || (R.K != SpirvResource::Buffer && R.K != SpirvResource::StorageTexture)) {
      fail("globallycoherent requires a writable buffer or storage texture", D->getLocation()); return true;
    }
    decorate(R.Variable, Decoration::Coherent);
  }
  decorate(R.Variable, Decoration::DescriptorSet, {R.Set});
  decorate(R.Variable, Decoration::Binding, {R.Binding});
  SpirvValue V{R.Variable, VariableType, StorageClass, T, true}; V.Resource = &R;
  Globals[D->getCanonicalDecl()] = V;
  Resources.push_back(std::move(Owned));
  return true;
}
bool ProphetSpirvEmitter::constantBuffer(const HLSLBufferDecl *D) {
  std::vector<uint32_t> Members;
  std::vector<const VarDecl *> Fields;
  std::vector<SpirvType> Physical;
  std::vector<uint32_t> Offsets;
  for (const Decl *Child : D->buffer_decls()) {
    if (const auto *V = dyn_cast<VarDecl>(Child)) {
      if (V->getStorageClass() == SC_Static ||
          V->getType().getAddressSpace() == LangAS::hlsl_groupshared) {
        global(V); continue;
      }
      if (resource(V)) continue;
      Fields.push_back(V); Members.push_back(type(V->getType(), SpirvLayout::Uniform));
      Physical.push_back(physicalType(V->getType(), SpirvLayout::Uniform));
      const auto *Pack = V->getAttr<HLSLPackOffsetAttr>();
      Offsets.push_back(Pack ? uint32_t(Pack->getSubcomponent() * 16 + Pack->getComponent() * 4) : ~0u);
    }
  }
  if (Fields.empty()) return true;
  uint32_t Set = 0, Slot = 0;
  if (!binding(D, Set, Slot)) return false;
  uint32_t T = structType(Members, SpirvLayout::Uniform, true, Physical, Offsets);
  if (!T) return false;
  uint32_t V = globalVariable(T, StorageClass::Uniform, D->getName());
  if (ObjectMode) ObjectResourceOwners[V] = D;
  decorate(V, Decoration::DescriptorSet, {Set}); decorate(V, Decoration::Binding, {Slot});
  for (uint32_t I = 0; I < Fields.size(); ++I) {
    SpirvValue Field{V, Members[I], StorageClass::Uniform, Fields[I]->getType(), true};
    Field.Path = {I}; Globals[Fields[I]->getCanonicalDecl()] = Field;
    Debug.string(OpMemberName, {T, I}, Fields[I]->getName());
  }
  return Diagnostic.empty();
}
bool ProphetSpirvEmitter::global(const VarDecl *D) {
  if (isRayQuery(D->getType())) return fail("RayQuery must be a function-local variable", D->getLocation());
  if (shaderAttribute(D, "table"))
    return true;
  if (shaderAttribute(D, "resource")) {
    if (resource(D)) return Diagnostic.empty();
    return fail("shader::resource requires a shader descriptor resource", D->getLocation());
  }
  // Sema associates these declarations with an implicit $Globals buffer at
  // end-of-translation-unit. Its buffer_decls() supplies the actual members.
  if (D->getType().getAddressSpace() == LangAS::hlsl_constant) return true;
  if (D->getType()->isFunctionPointerType())
    return fail("runtime call target must be statically known", D->getLocation());
  if (needsDestruction(D->getType()))
    return fail("global shader objects must have trivial destruction", D->getLocation());
  if (D->getType().isConstQualified() && D->hasInit()) return true;
  if (resource(D)) return Diagnostic.empty();
  if (!semantic(D).empty() || hasShaderAnnotation(D, "prophet.global_invocation_id_x")) {
    GlobalSemanticDecls.push_back(D); return true;
  }
  uint32_t Storage = hasShaderAnnotation(D, "prophet.groupshared") ||
      D->getType().getAddressSpace() == LangAS::hlsl_groupshared
      ? StorageClass::Workgroup : StorageClass::Private;
  if (TaskPayloadDecls.contains(D->getCanonicalDecl())) Storage = StorageClass::TaskPayloadWorkgroupEXT;
  uint32_t T = type(D->getType());
  if (!T) return false;
  uint32_t Init = 0;
  if (Storage == StorageClass::Private) {
    if (D->hasInit()) {
      Init = constantExpr(D->getInit()).Id;
      if (!Init) {
        const Expr *E = D->getInit()->IgnoreParenImpCasts();
        const auto *C = dyn_cast<CXXConstructExpr>(E);
        if (!isa<ImplicitValueInitExpr>(E) && !(C && C->getConstructor()->isTrivial() && !C->getNumArgs()))
          return fail("global shader initializers must be constant; perform runtime initialization in the entry", D->getLocation());
      }
    }
    if (!Init) Init = zero(T);
  }
  uint32_t V = globalVariable(T, Storage, D->getName(), Init);
  Globals[D->getCanonicalDecl()] = {V, T, Storage, D->getType(), true};
  return Diagnostic.empty();
}
SpirvValue ProphetSpirvEmitter::resourceIndex(SpirvValue Object, SpirvValue Index,
                                             QualType T) {
  SpirvResource *R = Object.Resource;
  if (!R) return {};
  UsedGlobals.insert(R->Variable); Index = load(Index);
  if (R->Count > 1 && !Object.DescriptorSelected) {
    uint32_t P = valueOp(OpAccessChain, pointerType(R->Type, Object.Storage),
                        {R->Variable, Index.Id});
    bool ConstantIndex = false;
    for (const auto &Entry : ConstantKeys) ConstantIndex |= Entry.second == Index.Id;
    // This lowering adds nonuniform support for storage buffer descriptor
    // arrays. Other resource kinds keep their existing stage/feature contract.
    if (!ConstantIndex && R->K == SpirvResource::Buffer && !R->Uniform) {
      extension("SPV_EXT_descriptor_indexing");
      capability(Capability::ShaderNonUniform);
      capability(Capability::StorageBufferArrayNonUniformIndexing);
      decorate(Index.Id, Decoration::NonUniform);
      decorate(P, Decoration::NonUniform);
    }
    Object.Id = P; Object.Type = R->Type; Object.Address = true; Object.DescriptorSelected = true;
    return Object;
  }
  if (R->K == SpirvResource::Buffer) {
    uint32_t Member = R->CppRecord ? uint32_t(Types[R->Type].Members.size() - 1) : 0;
    uint32_t P = valueOp(OpAccessChain, pointerType(R->Element, Object.Storage),
        {Object.Id, constant(integerType(), Member), Index.Id});
    return {P, R->Element, Object.Storage, T.getNonReferenceType(), true};
  }
  if (R->K == SpirvResource::StorageTexture) {
    Object.ImageCoordinate = Index.Id; Object.ASTType = T.getNonReferenceType();
    return Object;
  }
  fail("this resource does not support indexed element access"); return {};
}
}
