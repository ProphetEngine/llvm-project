//===-- ProphetSpirvEmitter.Interfaces.cpp - Addressable stage values -----===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

QualType ProphetSpirvEmitter::componentType(QualType T, uint32_t Index) {
  T = T.getNonReferenceType();
  if (const auto *A = Ctx.getAsArrayType(T)) return A->getElementType();
  if (const auto *V = T->getAs<VectorType>()) return V->getElementType();
  if (const auto *R = T->getAsRecordDecl()) {
    if (const auto *C = dyn_cast<CXXRecordDecl>(R)) {
      for (const auto &Base : C->bases()) {
        if (!Index) return Base.getType(); --Index;
      }
    }
    for (const FieldDecl *F : R->fields()) {
      if (!Index) return F->getType(); --Index;
    }
  }
  return {};
}
SpirvValue ProphetSpirvEmitter::interfaceReference(const ValueDecl *D, QualType T,
                                                  uint32_t Storage) {
  SpirvValue V{0, type(T), Storage, T, true}; V.Interface = D; return V;
}
SpirvValue ProphetSpirvEmitter::interfaceAddress(SpirvValue V) {
  auto Literal = [&](uint32_t Id) {
    for (const auto &[K, Value] : ConstantKeys)
      if (Value == Id && K[0] == OpConstant && K.size() >= 3) return K[2];
    return ~0u;
  };
  for (const auto &I : Interfaces) {
    if (I.Decl != V.Interface || I.Storage != V.Storage) continue;
    unsigned First = I.ArraySize && !I.Patch ? 1 : 0;
    if (V.InterfacePath.size() < First + I.Path.size()) continue;
    bool Match = true;
    for (size_t P = 0; P < I.Path.size(); ++P)
      Match &= Literal(V.InterfacePath[First + P]) == I.Path[P];
    if (!Match) continue;
    std::vector<uint32_t> Indices;
    if (First) Indices.push_back(V.InterfacePath[0]);
    uint32_t Type = I.Type;
    for (size_t P = First + I.Path.size(); P < V.InterfacePath.size(); ++P) {
      Indices.push_back(V.InterfacePath[P]);
      Type = Types[Type].K == SpirvType::Struct
          ? Types[Type].Members[Literal(V.InterfacePath[P])] : Types[Type].Element;
    }
    uint32_t Pointer = I.Variable;
    UsedGlobals.insert(Pointer);
    if (!Indices.empty()) {
      std::vector<uint32_t> Args{Pointer}; Args.insert(Args.end(), Indices.begin(), Indices.end());
      Pointer = valueOp(OpAccessChain, pointerType(Type, I.Storage), Args);
    }
    return {Pointer, Type, I.Storage, V.ASTType, true};
  }
  fail("shader stage subobject has no interface binding"); return {};
}
SpirvValue ProphetSpirvEmitter::loadInterface(SpirvValue V) {
  const auto Info = Types[V.Type];
  if (const auto *R = V.ASTType.getNonReferenceType()->getAsRecordDecl();
      R && !shaderAnnotation(R, "prophet.vector:").empty())
    return convert(load(interfaceAddress(V)), V.Type, V.ASTType);
  if (Info.K == SpirvType::Struct || Info.K == SpirvType::Array) {
    uint32_t Count = Info.K == SpirvType::Struct ? Info.Members.size() : Info.Count;
    std::vector<uint32_t> Values;
    for (uint32_t I = 0; I < Count; ++I) {
      uint32_t T = Info.K == SpirvType::Struct ? Info.Members[I] : Info.Element;
      Values.push_back(loadInterface(access(V, {constant(integerType(), I)}, componentType(V.ASTType, I), T)).Id);
    }
    return {composite(V.Type, Values), V.Type, 7, V.ASTType};
  }
  return convert(load(interfaceAddress(V)), V.Type, V.ASTType);
}
bool ProphetSpirvEmitter::storeInterface(SpirvValue Target, SpirvValue Value) {
  const auto Info = Types[Target.Type];
  if (const auto *R = Target.ASTType.getNonReferenceType()->getAsRecordDecl();
      R && !shaderAnnotation(R, "prophet.vector:").empty())
    return store(interfaceAddress(Target), Value);
  if (Info.K == SpirvType::Struct || Info.K == SpirvType::Array) {
    Value = convert(Value, Target.Type, Target.ASTType);
    uint32_t Count = Info.K == SpirvType::Struct ? Info.Members.size() : Info.Count;
    for (uint32_t I = 0; I < Count; ++I) {
      uint32_t T = Info.K == SpirvType::Struct ? Info.Members[I] : Info.Element;
      QualType AST = componentType(Target.ASTType, I);
      SpirvValue Part{valueOp(OpCompositeExtract, T, {Value.Id, I}), T, 7, AST};
      if (!storeInterface(access(Target, {constant(integerType(), I)}, AST, T), Part)) return false;
    }
    return true;
  }
  return store(interfaceAddress(Target), Value);
}
}
