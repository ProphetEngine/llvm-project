//===-- ProphetSpirvEmitter.DescriptorHeaps.cpp - Typed heap lookup ------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"
#include "prophet_shader_compiler.h"

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

SpirvValue ProphetSpirvEmitter::descriptorHeapCall(const CallExpr *E,
    SpirvValue Object, ArrayRef<SpirvValue> Args) {
  if (Environment < PROPHET_SHADER_ENVIRONMENT_VULKAN_1_3) {
    fail("typed descriptor heaps require target Vulkan 1.3 or later", E->getExprLoc()); return {};
  }
  const FunctionDecl *Function = E->getDirectCallee();
  SpirvValue Index;
  if (hasShaderAnnotation(Function, "prophet.descriptor_handle")) {
    if (!Args.empty() || !Object) {
      fail("DescriptorHandle lookup requires one typed handle receiver", E->getExprLoc()); return {};
    }
    Index = load(access(Object, {constant(integerType(), 0)}, Ctx.UnsignedIntTy));
  } else {
    if (Args.size() != 1) {
      fail("GetDescriptor<T> requires one logical uint heap index", E->getExprLoc()); return {};
    }
    Index = load(Args[0]);
  }
  if (!Index || Types[Index.Type].K != SpirvType::Int || Types[Index.Type].Bits != 32 || Types[Index.Type].Signed) {
    fail("descriptor heap indices must be unsigned 32-bit integers", E->getExprLoc()); return {};
  }

  QualType ResourceType = E->getType().getNonReferenceType();
  // Reuse the exact ordinary resource type/layout validation without emitting
  // a bound descriptor variable. The synthetic AST declaration owns type info
  // only; it is never a shader resource or a host-visible binding.
  auto *Declaration = VarDecl::Create(Ctx, Ctx.getTranslationUnitDecl(), E->getExprLoc(), E->getExprLoc(),
      &Ctx.Idents.get("__prophet_heap_view"), ResourceType,
      Ctx.getTrivialTypeSourceInfo(ResourceType), SC_None);
  SpirvResource *Resource = nullptr;
  if (!resource(Declaration, &Resource) || !Resource || !Diagnostic.empty()) {
    if (Diagnostic.empty()) fail("GetDescriptor<T> requires a buffer, texture, sampler or acceleration-structure resource type", E->getExprLoc());
    return {};
  }
  return descriptorHeapValue(Resource, ResourceType, Index);
}

SpirvValue ProphetSpirvEmitter::descriptorHeapValue(SpirvResource *Resource,
    QualType ResourceType, SpirvValue Index) {
  bool Sampler = Resource->K == SpirvResource::Sampler || Resource->K == SpirvResource::ComparisonSampler;
  unsigned HeapKind = Sampler ? 1 : 0;
  extension("SPV_KHR_untyped_pointers"); extension("SPV_EXT_descriptor_heap");
  capability(Capability::UntypedPointersKHR); capability(Capability::DescriptorHeapEXT);
  uint32_t UntypedPointer = internType(OpTypeUntypedPointerKHR, {StorageClass::UniformConstant}, {});
  if (!DescriptorHeaps[HeapKind]) {
    uint32_t Heap = id();
    GlobalVariables.emit(OpUntypedVariableKHR, {UntypedPointer, Heap, StorageClass::UniformConstant});
    decorate(Heap, Decoration::BuiltIn, {Sampler ? BuiltIn::SamplerHeapEXT : BuiltIn::ResourceHeapEXT});
    name(Heap, Sampler ? "__prophet_sampler_heap" : "__prophet_resource_heap");
    AllGlobals.insert(Heap); trackPointer(Heap, StorageClass::UniformConstant);
    DescriptorHeaps[HeapKind] = Heap;
    uint32_t Stride = id();
    Declarations.emit(OpSpecConstant, {integerType(), Stride, 16});
    decorate(Stride, Decoration::SpecId, {0x50520000u + HeapKind});
    name(Stride, Sampler ? "__prophet_sampler_heap_stride" : "__prophet_resource_heap_stride");
    DescriptorHeapStrides[HeapKind] = Stride;
  }
  uint32_t Heap = DescriptorHeaps[HeapKind];
  UsedGlobals.insert(Heap);
  uint32_t Storage = Resource->K == SpirvResource::Buffer
      ? (Resource->Uniform ? StorageClass::Uniform : StorageClass::StorageBuffer)
      : StorageClass::UniformConstant;
  uint32_t DescriptorType = Resource->K == SpirvResource::Buffer
      ? internType(OpTypeBufferEXT, {Storage}, {}) : Resource->Type;
  SpirvType ArrayInfo; ArrayInfo.K = SpirvType::RuntimeArray; ArrayInfo.Element = DescriptorType;
  uint32_t Array = internType(OpTypeRuntimeArray, {DescriptorType}, ArrayInfo);
  // Standard heap accesses are nonuniform by definition. Keep the logical
  // index as the access-chain index; only the client-selected stride is bytes.
  if (DescriptorHeapArrays.insert(Array).second)
    Decorations.emit(OpDecorateId, {Array, Decoration::ArrayStrideIdEXT, DescriptorHeapStrides[HeapKind]});
  uint32_t Pointer = valueOp(OpUntypedAccessChainKHR, UntypedPointer, {Array, Heap, Index.Id});
  if (Resource->K == SpirvResource::Buffer) {
    Pointer = valueOp(OpBufferPointerEXT, pointerType(Resource->Type, Storage), {Pointer});
    trackPointer(Pointer, Storage);
    if (Resource->ReadOnly && !Resource->Uniform) decorate(Pointer, Decoration::NonWritable);
    if (ResourceType->getAsCXXRecordDecl()->getName().contains("ByteAddress")) {
      extension("SPV_GOOGLE_user_type");
      Decorations.string(OpDecorateString, {Pointer, Decoration::UserTypeGOOGLE}, "ByteAddressBuffer");
    }
  }
  Resource->Variable = Pointer; Resource->Heap = Heap;
  SpirvValue Value{Pointer, Resource->Type, Storage, ResourceType, true};
  Value.Resource = Resource; Value.DescriptorSelected = true;
  Value.HeapIndex = Index.Id;
  return Value;
}
}
