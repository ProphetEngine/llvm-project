//===-- ProphetSpirvEmitter.Types.cpp - Shader types and values -----------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"
#include "clang/AST/RecordLayout.h"
#include "llvm/Support/MathExtras.h"
#include <algorithm>

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

uint32_t ProphetSpirvEmitter::internType(uint32_t Op, ArrayRef<uint32_t> Args,
                                         SpirvType Info) {
  std::vector<uint32_t> Key{Op};
  Key.insert(Key.end(), Args.begin(), Args.end());
  auto I = TypeKeys.find(Key);
  if (I != TypeKeys.end()) return I->second;
  uint32_t T = id();
  TypeKeys[Key] = T;
  Types[T] = std::move(Info);
  std::vector<uint32_t> Operands{T};
  Operands.insert(Operands.end(), Args.begin(), Args.end());
  Declarations.emit(Op, Operands);
  return T;
}
uint32_t ProphetSpirvEmitter::integerType(uint32_t Bits, bool Signed) {
  if (Bits == 64) capability(Capability::Int64);
  if (Bits == 16) capability(Capability::Int16);
  if (Bits == 8) capability(Capability::Int8);
  SpirvType T; T.K = SpirvType::Int; T.Bits = Bits; T.Signed = Signed;
  T.Size = T.Align = Bits / 8;
  return internType(OpTypeInt, {Bits, uint32_t(Signed)}, T);
}
uint32_t ProphetSpirvEmitter::floatType(uint32_t Bits) {
  if (Bits == 64) capability(Capability::Float64);
  if (Bits == 16) capability(Capability::Float16);
  SpirvType T; T.K = SpirvType::Float; T.Bits = Bits;
  T.Size = T.Align = Bits / 8;
  return internType(OpTypeFloat, {Bits}, T);
}
uint32_t ProphetSpirvEmitter::boolType() {
  SpirvType T; T.K = SpirvType::Bool; T.Size = T.Align = 4;
  return internType(OpTypeBool, {}, T);
}
uint32_t ProphetSpirvEmitter::voidType() {
  return internType(OpTypeVoid, {}, {});
}
uint32_t ProphetSpirvEmitter::vectorType(uint32_t Element, uint32_t Count) {
  if (Count == 1) return Element;
  if (Count < 2 || Count > 4) {
    fail("shader vectors must have two to four elements");
    return 0;
  }
  SpirvType T; T.K = SpirvType::Vector; T.Element = Element; T.Count = Count;
  T.Size = Types[Element].Size * Count;
  T.Align = Types[Element].Align * (Count == 3 ? 4 : Count);
  return internType(OpTypeVector, {Element, Count}, T);
}
uint32_t ProphetSpirvEmitter::arrayType(uint32_t Element, uint32_t Count,
                                       SpirvLayout Layout, bool Runtime, const SpirvType *ElementLayout) {
  SpirvType Physical = ElementLayout ? *ElementLayout : physicalType(Element, Layout);
  uint32_t Align = Physical.Align;
  if (Layout == SpirvLayout::Uniform) Align = std::max(16u, Align);
  uint32_t Stride = uint32_t(alignTo(Physical.Size, Align));
  uint32_t Op = Runtime ? OpTypeRuntimeArray : OpTypeArray;
  std::vector<uint32_t> Key{Op, Element, Count, Stride, uint32_t(Layout), uint32_t(Physical.RowMajor), Physical.MatrixStride};
  if (auto I = TypeKeys.find(Key); I != TypeKeys.end()) return I->second;
  uint32_t Length = Runtime ? 0 : constant(integerType(), Count);
  uint32_t T = id();
  TypeKeys[Key] = T;
  SpirvType Info; Info.K = Runtime ? SpirvType::RuntimeArray : SpirvType::Array;
  Info.Element = Element; Info.Count = Count; Info.Align = Align;
  Info.RowMajor = Physical.RowMajor; Info.MatrixStride = Physical.MatrixStride;
  Info.Size = Runtime ? 0 : Stride * Count;
  Types[T] = Info;
  if (Runtime) Declarations.emit(Op, {T, Element});
  else Declarations.emit(Op, {T, Element, Length});
  if (Layout != SpirvLayout::Natural) decorate(T, Decoration::ArrayStride, {Stride});
  return T;
}
uint32_t ProphetSpirvEmitter::structType(ArrayRef<uint32_t> Members,
                                        SpirvLayout Layout, bool Block, ArrayRef<SpirvType> MemberLayouts,
                                        ArrayRef<uint32_t> ExplicitOffsets) {
  std::vector<SpirvType> Physical;
  for (uint32_t I = 0; I < Members.size(); ++I)
    Physical.push_back(MemberLayouts.empty() ? physicalType(Members[I], Layout) : MemberLayouts[I]);
  // Layout and Block membership are part of type identity, not just decoration.
  std::vector<uint32_t> Key{OpTypeStruct, ~0u, uint32_t(Layout), uint32_t(Block)};
  Key.insert(Key.end(), Members.begin(), Members.end());
  Key.push_back(~0u); Key.insert(Key.end(), ExplicitOffsets.begin(), ExplicitOffsets.end());
  for (const auto &M : Physical) {
    Key.push_back(M.Size); Key.push_back(M.Align);
    Key.push_back(M.RowMajor); Key.push_back(M.MatrixStride);
  }
  if (auto I = TypeKeys.find(Key); I != TypeKeys.end()) return I->second;
  uint32_t T = id();
  TypeKeys[Key] = T;
  SpirvType Info; Info.K = SpirvType::Struct;
  Info.Align = Layout == SpirvLayout::Uniform ? 16 : 1;
  for (uint32_t I = 0; I < Members.size(); ++I) {
    uint32_t M = Members[I];
    const auto &MT = Physical[I];
    Info.Align = std::max(Info.Align, MT.Align);
    uint32_t Offset = uint32_t(alignTo(Info.Size, MT.Align));
    if (!ExplicitOffsets.empty() && ExplicitOffsets[I] != ~0u) {
      Offset = ExplicitOffsets[I];
      uint32_t Alignment = Types[M].K == SpirvType::Vector ? Types[Types[M].Element].Align : MT.Align;
      if (Offset % Alignment) { fail("packoffset does not satisfy the shader member alignment"); return 0; }
      for (uint32_t J = 0; J < I; ++J)
        if (Offset < Info.Offsets[J] + Physical[J].Size && Info.Offsets[J] < Offset + MT.Size) {
          fail("constant buffer members have overlapping packoffset ranges"); return 0;
        }
    }
    Info.Offsets.push_back(Offset);
    Info.Members.push_back(M);
    Info.Size = std::max(Info.Size, Offset + MT.Size);
  }
  Info.Size = uint32_t(alignTo(Info.Size, Info.Align));
  Types[T] = Info;
  std::vector<uint32_t> Args{T};
  Args.insert(Args.end(), Members.begin(), Members.end());
  Declarations.emit(OpTypeStruct, Args);
  if (Block) decorate(T, Decoration::Block);
  for (uint32_t I = 0; Layout != SpirvLayout::Natural && I < Members.size(); ++I) {
    memberDecorate(T, I, Decoration::Offset, {Info.Offsets[I]});
    uint32_t M = Members[I];
    while (Types[M].K == SpirvType::Array) M = Types[M].Element;
    if (Types[M].K == SpirvType::Matrix && Types[M].NativeMatrix) {
      const auto &Matrix = Physical[I];
      memberDecorate(T, I, Matrix.RowMajor ? Decoration::RowMajor : Decoration::ColMajor);
      memberDecorate(T, I, Decoration::MatrixStride, {Matrix.MatrixStride});
    }
  }
  return T;
}
static bool isRowMajor(QualType T) {
  for (const clang::Type *Raw = T.getNonReferenceType().getTypePtr(); const auto *A = dyn_cast<AttributedType>(Raw); Raw = A->getModifiedType().getTypePtr()) {
    if (A->getAttrKind() == attr::HLSLRowMajor) return true;
    if (A->getAttrKind() == attr::HLSLColumnMajor) return false;
  }
  return false;
}
SpirvType ProphetSpirvEmitter::physicalType(uint32_t T, SpirvLayout Layout, bool RowMajor) {
  SpirvType Info = Types[T];
  if (Info.K != SpirvType::Matrix || !Info.NativeMatrix) return Info;
  Info.RowMajor = Layout != SpirvLayout::Natural && RowMajor;
  const auto Column = Types[Info.Element], Scalar = Types[Column.Element];
  uint32_t Components = Info.RowMajor ? Info.Count : Column.Count;
  Info.Align = Scalar.Align * (Components == 3 ? 4 : Components);
  if (Layout == SpirvLayout::Uniform) Info.Align = std::max(16u, Info.Align);
  Info.MatrixStride = uint32_t(alignTo(Scalar.Size * Components, Info.Align));
  Info.Size = Info.MatrixStride * (Info.RowMajor ? Column.Count : Info.Count);
  return Info;
}
SpirvType ProphetSpirvEmitter::physicalType(QualType T, SpirvLayout Layout) {
  return physicalType(type(T, Layout), Layout, isRowMajor(T));
}
uint32_t ProphetSpirvEmitter::pointerType(uint32_t T, uint32_t Storage) {
  return internType(OpTypePointer, {Storage, T}, {});
}
uint32_t ProphetSpirvEmitter::type(QualType T, SpirvLayout Layout) {
  if (T.isNull()) return 0;
  bool RowMajor = Layout != SpirvLayout::Natural && isRowMajor(T);
  QualType Original = T.getNonReferenceType().getUnqualifiedType();
  T = Original.getCanonicalType();
  auto Key = std::make_tuple(T.getAsOpaquePtr(), Layout, RowMajor);
  if (auto I = ASTTypes.find(Key); I != ASTTypes.end()) return I->second;
  uint32_t Result = 0;
  if (T->isVoidType()) Result = voidType();
  else if (T->isBooleanType()) Result = Layout == SpirvLayout::Natural ? boolType() : integerType();
  else if (T->isIntegerType() || T->isEnumeralType())
    Result = integerType(uint32_t(Ctx.getTypeSize(T)), T->isSignedIntegerOrEnumerationType());
  else if (T->isRealFloatingType()) Result = floatType(uint32_t(Ctx.getTypeSize(T)));
  else if (const auto *V = T->getAs<VectorType>())
    Result = vectorType(type(V->getElementType(), Layout), V->getNumElements());
  else if (const auto *A = Ctx.getAsConstantArrayType(Original)) {
    uint32_t Element = type(A->getElementType(), Layout);
    SpirvType Physical = physicalType(A->getElementType(), Layout);
    Result = arrayType(Element, uint32_t(A->getSize().getLimitedValue()), Layout, false, &Physical);
  } else if (const auto *M = T->getAs<ConstantMatrixType>()) {
    uint32_t Scalar = type(M->getElementType(), Layout);
    uint32_t Column = vectorType(Scalar, M->getNumRows());
    SpirvType Info; Info.K = SpirvType::Matrix; Info.Element = Column;
    Info.Rows = M->getNumRows(); Info.Count = M->getNumColumns();
    Info.NativeMatrix = Types[Scalar].K == SpirvType::Float &&
                        Info.Rows > 1 && Info.Count > 1;
    // Emulated matrices use a physical row array when row_major is requested.
    // Native SPIR-V matrices retain logical columns and use member decorations.
    Info.RowMajor = !Info.NativeMatrix && RowMajor;
    if (Info.RowMajor) Info.Element = vectorType(Scalar, Info.Count);
    Info.Align = Types[Info.Element].Align;
    if (Layout == SpirvLayout::Uniform) Info.Align = std::max(16u, Info.Align);
    Info.MatrixStride = uint32_t(alignTo(Types[Info.Element].Size, Info.Align));
    Info.Size = Info.MatrixStride * (Info.RowMajor ? Info.Rows : Info.Count);
    if (Info.NativeMatrix) {
      capability(Capability::Matrix);
      Result = internType(OpTypeMatrix, {Column, Info.Count}, Info);
    } else {
      // OpTypeMatrix requires floating vectors and at least two columns.
      // Keep the logical matrix shape while representing other HLSL matrices
      // as arrays of columns (a scalar column for one-row matrices).
      std::vector<uint32_t> MatrixKey{OpTypeArray, ~0u, Info.Element, Info.Count, Info.Rows,
                                      uint32_t(Layout), Info.MatrixStride, uint32_t(Info.RowMajor)};
      if (auto I = TypeKeys.find(MatrixKey); I != TypeKeys.end()) Result = I->second;
      else {
        uint32_t Length = constant(integerType(), Info.RowMajor ? Info.Rows : Info.Count);
        Result = id(); TypeKeys[MatrixKey] = Result; Types[Result] = Info;
        Declarations.emit(OpTypeArray, {Result, Info.Element, Length});
        if (Layout != SpirvLayout::Natural)
          decorate(Result, Decoration::ArrayStride, {Info.MatrixStride});
      }
    }
  } else if (const auto *R = T->getAsRecordDecl()) {
    if (!R->getDefinition()) { fail("shader record has no definition", R->getLocation()); return 0; }
    if (R->isUnion()) { fail("runtime union objects require an explicit shader representation; unions are supported in constant evaluation only", R->getLocation()); return 0; }
    std::vector<uint32_t> Members;
    std::vector<SpirvType> Physical;
    if (const auto *C = dyn_cast<CXXRecordDecl>(R)) {
      if (C->isPolymorphic() || C->getNumVBases()) {
        fail("virtual shader objects have no SPIR-V representation", C->getLocation()); return 0;
      }
      for (const CXXBaseSpecifier &B : C->bases()) {
        Members.push_back(type(B.getType(), Layout)); Physical.push_back(physicalType(B.getType(), Layout));
      }
    }
    for (const FieldDecl *F : R->fields()) {
      if (F->isBitField() || F->getType()->isPointerType() || F->getType()->isReferenceType()) {
        fail("stored shader pointer/reference and bitfield members require an explicit buffer ABI", F->getLocation());
        return 0;
      }
      Members.push_back(type(F->getType(), Layout));
      Physical.push_back(physicalType(F->getType(), Layout));
    }
    if (!Diagnostic.empty()) return 0;
    Result = structType(Members, Layout, false, Physical);
    name(Result, R->getName());
    uint32_t I = 0;
    if (const auto *C = dyn_cast<CXXRecordDecl>(R)) I = C->getNumBases();
    for (const FieldDecl *F : R->fields())
      Debug.string(OpMemberName, {Result, I++}, F->getName());
  } else {
    fail("AST type has no shader SPIR-V representation: " + T.getAsString());
  }
  if (Result) ASTTypes[Key] = Result;
  return Result;
}
uint32_t ProphetSpirvEmitter::intrinsicType(QualType T) {
  T = T.getNonReferenceType();
  if (const auto *R = T->getAsRecordDecl()) {
    StringRef Width = shaderAnnotation(R, "prophet.vector:");
    uint32_t Count = 0;
    if (!Width.empty() && !Width.getAsInteger(10, Count) && !R->field_empty())
      return vectorType(type((*R->field_begin())->getType()), Count);
  }
  return type(T);
}
uint32_t ProphetSpirvEmitter::constant(uint32_t T, uint64_t Bits) {
  if (Types[T].K == SpirvType::Bool) return constantBool(Bits != 0);
  std::vector<uint32_t> Key{OpConstant, T, uint32_t(Bits)};
  if (Types[T].Bits == 64) Key.push_back(uint32_t(Bits >> 32));
  if (auto I = ConstantKeys.find(Key); I != ConstantKeys.end()) return I->second;
  uint32_t V = id(); ConstantKeys[Key] = V;
  std::vector<uint32_t> Args{T, V};
  Args.insert(Args.end(), Key.begin() + 2, Key.end());
  Declarations.emit(OpConstant, Args);
  return V;
}
uint32_t ProphetSpirvEmitter::constantBool(bool V) {
  uint32_t T = boolType(), Op = V ? OpConstantTrue : OpConstantFalse;
  std::vector<uint32_t> Key{Op, T};
  if (auto I = ConstantKeys.find(Key); I != ConstantKeys.end()) return I->second;
  uint32_t R = id(); ConstantKeys[Key] = R;
  Declarations.emit(Op, {T, R}); return R;
}
uint32_t ProphetSpirvEmitter::zero(uint32_t T) {
  std::vector<uint32_t> Key{OpConstantNull, T};
  if (auto I = ConstantKeys.find(Key); I != ConstantKeys.end()) return I->second;
  uint32_t R = id(); ConstantKeys[Key] = R;
  Declarations.emit(OpConstantNull, {T, R}); return R;
}
uint32_t ProphetSpirvEmitter::composite(uint32_t T, ArrayRef<uint32_t> Values, bool Constant) {
  // HLSL one-component vectors have scalar SPIR-V representation.
  if (Types[T].K == SpirvType::Bool || Types[T].K == SpirvType::Int ||
      Types[T].K == SpirvType::Float) {
    if (Values.size() == 1) return Values.front();
    fail("scalar shader construction requires one component"); return 0;
  }
  if (!Constant) return valueOp(OpCompositeConstruct, T, Values);
  std::vector<uint32_t> Key{OpConstantComposite, T};
  Key.insert(Key.end(), Values.begin(), Values.end());
  if (auto I = ConstantKeys.find(Key); I != ConstantKeys.end()) return I->second;
  uint32_t R = id(); ConstantKeys[Key] = R;
  std::vector<uint32_t> Args{T, R}; Args.insert(Args.end(), Values.begin(), Values.end());
  Declarations.emit(OpConstantComposite, Args); return R;
}
uint32_t ProphetSpirvEmitter::constantValue(QualType T, const APValue &V) {
  uint32_t ST = type(T);
  if (V.isInt()) return constant(ST, V.getInt().getLimitedValue());
  if (V.isFloat()) return constant(ST, V.getFloat().bitcastToAPInt().getLimitedValue());
  std::vector<uint32_t> Values;
  if (V.isVector()) {
    QualType Element = T->getAs<VectorType>()->getElementType();
    for (unsigned I = 0; I < V.getVectorLength(); ++I)
      Values.push_back(constantValue(Element, V.getVectorElt(I)));
  } else if (V.isArray()) {
    QualType Element = Ctx.getAsArrayType(T)->getElementType();
    for (unsigned I = 0; I < V.getArraySize(); ++I)
      Values.push_back(constantValue(Element, I < V.getArrayInitializedElts()
          ? V.getArrayInitializedElt(I) : V.getArrayFiller()));
  } else if (V.isStruct()) {
    const RecordDecl *R = T->getAsRecordDecl();
    unsigned I = 0;
    if (const auto *C = dyn_cast<CXXRecordDecl>(R))
      for (const auto &B : C->bases()) Values.push_back(constantValue(B.getType(), V.getStructBase(I++)));
    I = 0;
    for (const FieldDecl *F : R->fields()) Values.push_back(constantValue(F->getType(), V.getStructField(I++)));
  } else return 0;
  if (llvm::is_contained(Values, 0u)) return 0;
  return composite(ST, Values, true);
}
SpirvValue ProphetSpirvEmitter::constantExpr(const Expr *E) {
  if (E->getType()->isPointerType() || E->getType()->isReferenceType() ||
      E->isValueDependent()) return {};
  if (const auto *Record = E->getType()->getAsCXXRecordDecl(); Record && Record->isLambda()) return {};
  Expr::EvalResult R;
  if (!E->EvaluateAsRValue(R, Ctx) || R.HasSideEffects ||
      (!R.Val.isInt() && !R.Val.isFloat() && !R.Val.isVector() && !R.Val.isArray() && !R.Val.isStruct()))
    return {};
  uint32_t V = constantValue(E->getType(), R.Val);
  return {V, type(E->getType()), 7, E->getType()};
}

SpirvValue ProphetSpirvEmitter::temporary(QualType T, StringRef Name) {
  return temporaryType(type(T), T.getNonReferenceType(), Name);
}
SpirvValue ProphetSpirvEmitter::temporaryType(uint32_t T, QualType AST, StringRef Name) {
  uint32_t V = id();
  LocalVariables.emit(OpVariable, {pointerType(T, StorageClass::Function), V, StorageClass::Function});
  name(V, Name);
  trackPointer(V, StorageClass::Function);
  return {V, T, StorageClass::Function, AST, true};
}
SpirvValue ProphetSpirvEmitter::access(SpirvValue Base, ArrayRef<uint32_t> Indices,
                                       QualType Result, uint32_t ResultType) {
  if (Base.Interface) {
    Base.Type = ResultType ? ResultType : type(Result); Base.ASTType = Result;
    Base.InterfacePath.insert(Base.InterfacePath.end(), Indices.begin(), Indices.end());
    return Base;
  }
  Base = address(Base);
  if (!Base.Address) { fail("subobject access requires an address"); return {}; }
  uint32_t T = ResultType ? ResultType : type(Result);
  std::vector<uint32_t> Args{Base.Id}; Args.insert(Args.end(), Indices.begin(), Indices.end());
  uint32_t V = valueOp(OpAccessChain, pointerType(T, Base.Storage), Args);
  return {V, T, Base.Storage, Result, true};
}
SpirvValue ProphetSpirvEmitter::extract(SpirvValue Base, ArrayRef<uint32_t> Indices, QualType T) {
  Base = load(Base);
  std::vector<uint32_t> Args{Base.Id}; Args.insert(Args.end(), Indices.begin(), Indices.end());
  return {valueOp(OpCompositeExtract, type(T), Args), type(T), 7, T};
}
uint32_t ProphetSpirvEmitter::repack(uint32_t Value, uint32_t From, uint32_t To) {
  if (From == To) return Value;
  const auto F = Types[From], T = Types[To];
  if (F.K == SpirvType::Matrix && T.K == SpirvType::Matrix) {
    bool FR = !F.NativeMatrix && F.RowMajor, TR = !T.NativeMatrix && T.RowMajor;
    uint32_t FS = matrixScalarType(From), TS = matrixScalarType(To);
    std::vector<uint32_t> Outer;
    for (uint32_t I = 0; I < (TR ? T.Rows : T.Count); ++I) {
      std::vector<uint32_t> Inner;
      for (uint32_t J = 0; J < (TR ? T.Count : T.Rows); ++J) {
        uint32_t Row = TR ? I : J, Col = TR ? J : I;
        if (Row >= F.Rows || Col >= F.Count) { Inner.push_back(zero(TS)); continue; }
        std::vector<uint32_t> Indices{Value, FR ? Row : Col};
        if ((FR ? F.Count : F.Rows) > 1) Indices.push_back(FR ? Col : Row);
        uint32_t Part = valueOp(OpCompositeExtract, FS, Indices);
        Inner.push_back(convert({Part, FS}, TS).Id);
      }
      Outer.push_back(composite(T.Element, Inner));
    }
    return composite(To, Outer);
  }
  if (F.K == SpirvType::Struct || F.K == SpirvType::Array || F.K == SpirvType::Matrix) {
    std::vector<uint32_t> Values;
    unsigned Count = F.K == SpirvType::Struct ? F.Members.size() : F.Count;
    for (unsigned I = 0; I < Count; ++I) {
      uint32_t FT = F.K == SpirvType::Struct ? F.Members[I] : F.Element;
      uint32_t TT = T.K == SpirvType::Struct ? T.Members[I] : T.Element;
      Values.push_back(repack(valueOp(OpCompositeExtract, FT, {Value, I}), FT, TT));
    }
    return composite(To, Values);
  }
  return convert({Value, From}, To).Id;
}
SpirvValue ProphetSpirvEmitter::address(SpirvValue V) {
  if (AllGlobals.contains(V.Id)) UsedGlobals.insert(V.Id);
  if (!V.Path.empty()) {
    std::vector<uint32_t> Args{V.Id};
    for (uint32_t P : V.Path) Args.push_back(constant(integerType(), P));
    V.Id = valueOp(OpAccessChain, pointerType(V.Type, V.Storage), Args);
    V.Path.clear();
  }
  return V;
}
SpirvValue ProphetSpirvEmitter::load(SpirvValue V) {
  if (!V.MatrixComponents.empty()) {
    auto Components = std::move(V.MatrixComponents); V.MatrixComponents.clear();
    auto Swizzle = std::move(V.Swizzle); V.Swizzle.clear();
    uint32_t Scalar = matrixScalarType(V.Type);
    QualType Element = V.ASTType;
    if (const auto *Vector = Element->getAs<VectorType>()) Element = Vector->getElementType();
    std::vector<uint32_t> Values;
    if (!V.Address) {
      SpirvValue Copy = temporaryType(V.Type); store(Copy, V); V = Copy;
    }
    for (auto [Row, Column] : Components)
      Values.push_back(load(matrixAddress(V, Row, Column, Element)).Id);
    uint32_t T = vectorType(Scalar, Values.size());
    SpirvValue Result{composite(T, Values), T, 7, V.ASTType};
    Result.Swizzle = std::move(Swizzle); return load(Result);
  }
  if (V.Interface) {
    auto Swizzle = V.Swizzle; V.Swizzle.clear();
    V = loadInterface(V); V.Swizzle = std::move(Swizzle);
  }
  V = address(V);
  // A storage buffer value is a descriptor receiver, not the contents of its
  // block. Builtin resource copy construction and by-value helper arguments
  // retain this provenance; only resourceIndex produces a loadable element.
  // Loading the block here would also attempt to load its runtime-sized array.
  if (V.Resource && V.Resource->K == SpirvResource::Buffer) {
    UsedGlobals.insert(V.Resource->Variable);
    return V;
  }
  if (V.Resource && V.Resource->K != SpirvResource::Buffer) {
    UsedGlobals.insert(V.Resource->Variable);
    if (V.Address) V.Id = valueOp(OpLoad, V.Resource->Type, {V.Id});
    V.Type = V.Resource->Type; V.Address = false;
    if (V.ImageCoordinate) {
      uint32_t Color = vectorType(V.Resource->Element, 4);
      uint32_t Value = valueOp(OpImageRead, Color, {V.Id, V.ImageCoordinate});
      SpirvValue Result{Value, Color, 7, V.ASTType};
      return convert(Result, type(V.ASTType), V.ASTType);
    }
    return V;
  }
  if (V.Address) {
    if (AllGlobals.contains(V.Id)) UsedGlobals.insert(V.Id);
    V.Id = valueOp(OpLoad, V.Type, {V.Id}); V.Address = false;
  }
  if (!V.Swizzle.empty()) {
    if (Types[V.Type].K != SpirvType::Vector) {
      if (V.Swizzle.size() > 1) {
        uint32_t T = vectorType(V.Type, V.Swizzle.size());
        V.Id = composite(T, std::vector<uint32_t>(V.Swizzle.size(), V.Id)); V.Type = T;
      }
      V.Swizzle.clear(); return V;
    }
    if (V.Swizzle.size() == 1) {
      V.Id = valueOp(OpCompositeExtract, Types[V.Type].Element, {V.Id, V.Swizzle[0]});
      V.Type = Types[V.Type].Element;
    } else {
      uint32_t T = vectorType(Types[V.Type].Element, V.Swizzle.size());
      std::vector<uint32_t> Args{V.Id, V.Id};
      Args.insert(Args.end(), V.Swizzle.begin(), V.Swizzle.end());
      V.Id = valueOp(OpVectorShuffle, T, Args); V.Type = T;
    }
    V.Swizzle.clear();
  }
  return V;
}
bool ProphetSpirvEmitter::store(SpirvValue Target, SpirvValue Value) {
  if (Target.Storage == StorageClass::Input || Target.Storage == StorageClass::Uniform)
    return fail("cannot write a read-only shader input or constant buffer");
  for (const auto &R : Resources)
    if (R->ReadOnly && pointerRoot(Target.Id) == R->Variable)
      return fail("cannot write a read-only shader resource");
  Target = address(Target);
  if (!Target || !Value) return false;
  if (!Target.Address && !Target.Interface && !Target.Resource)
    return fail("assignment target has no shader address");
  if (!Target.MatrixComponents.empty()) {
    auto Components = std::move(Target.MatrixComponents); Target.MatrixComponents.clear();
    auto Swizzle = std::move(Target.Swizzle); Target.Swizzle.clear();
    Value = load(Value);
    uint32_t Scalar = matrixScalarType(Target.Type);
    QualType Element = Target.ASTType;
    if (const auto *Vector = Element->getAs<VectorType>()) Element = Vector->getElementType();
    unsigned Count = Swizzle.empty() ? Components.size() : Swizzle.size();
    for (unsigned I = 0; I < Count; ++I) {
      unsigned Index = Swizzle.empty() ? I : Swizzle[I];
      SpirvValue Part = Value;
      if (Types[Value.Type].K == SpirvType::Vector)
        Part = {valueOp(OpCompositeExtract, Types[Value.Type].Element, {Value.Id, I}), Types[Value.Type].Element};
      auto [Row, Column] = Components[Index];
      if (!store(matrixAddress(Target, Row, Column, Element), convert(Part, Scalar))) return false;
    }
    return true;
  }
  if (Target.Resource && Target.ImageCoordinate) {
    auto *R = Target.Resource;
    if (R->ReadOnly) return fail("cannot write a read-only shader resource");
    SpirvValue ImageObject = Target; ImageObject.ImageCoordinate = 0;
    uint32_t Image = load(ImageObject).Id;
    Value = load(Value);
    Body.emit(OpImageWrite, {Image, Target.ImageCoordinate, Value.Id});
    return true;
  }
  if (!Target.Address) return fail("assignment target has no writable address");
  if (!Target.Swizzle.empty()) {
    auto Swizzle = Target.Swizzle; Target.Swizzle.clear();
    if (Types[Target.Type].K != SpirvType::Vector) return store(Target, Value);
    SpirvValue Original = load(Target); Value = load(Value);
    for (unsigned I = 0; I < Swizzle.size(); ++I) {
      uint32_t Element = Types[Target.Type].Element;
      uint32_t Part = Swizzle.size() == 1 ? Value.Id
          : valueOp(OpCompositeExtract, Element, {Value.Id, I});
      Original.Id = valueOp(OpCompositeInsert, Target.Type, {Part, Original.Id, Swizzle[I]});
    }
    return store(Target, Original);
  }
  if (Target.Interface) return storeInterface(Target, Value);
  Value = convert(Value, Target.Type, Target.ASTType);
  if (!Value) return false;
  Body.emit(OpStore, {Target.Id, Value.Id});
  trackStore(Target.Id, Value.Id);
  return true;
}
SpirvValue ProphetSpirvEmitter::convert(SpirvValue V, uint32_t T, QualType AST) {
  V = load(V);
  if (!V || !T) return {};
  if (V.Type == T) { V.ASTType = AST.isNull() ? V.ASTType : AST; return V; }
  SpirvType From = Types[V.Type], To = Types[T];
  // SDK padded vectors retain their host POD ABI in buffers and use vectors
  // for shader interfaces and native operations. Padding is never a channel.
  if ((From.K == SpirvType::Struct && To.K == SpirvType::Vector) ||
      (From.K == SpirvType::Vector && To.K == SpirvType::Struct)) {
    uint32_t Count = To.K == SpirvType::Struct ? To.Members.size() : To.Count;
    std::vector<uint32_t> Elements;
    for (uint32_t I = 0; I < Count; ++I) {
      uint32_t Destination = To.K == SpirvType::Struct ? To.Members[I] : To.Element;
      if (I >= (From.K == SpirvType::Struct ? From.Members.size() : From.Count)) {
        Elements.push_back(zero(Destination)); continue;
      }
      uint32_t Source = From.K == SpirvType::Struct ? From.Members[I] : From.Element;
      Elements.push_back(convert({valueOp(OpCompositeExtract, Source, {V.Id, I}), Source}, Destination).Id);
    }
    return {composite(T, Elements), T, 7, AST};
  }
  if ((To.K == SpirvType::Struct || To.K == SpirvType::Array) &&
      (From.K == SpirvType::Bool || From.K == SpirvType::Int || From.K == SpirvType::Float)) {
    unsigned Count = To.K == SpirvType::Struct ? To.Members.size() : To.Count;
    std::vector<uint32_t> Values;
    for (unsigned I = 0; I < Count; ++I)
      Values.push_back(convert(V, To.K == SpirvType::Struct ? To.Members[I] : To.Element).Id);
    return {composite(T, Values), T, 7, AST};
  }
  if (To.K == SpirvType::Matrix && From.K != SpirvType::Matrix) {
    SpirvValue Column = convert(V, To.Element);
    return {composite(T, std::vector<uint32_t>(To.Count, Column.Id)), T, 7, AST};
  }
  if (To.K == SpirvType::Vector && From.K != SpirvType::Vector) {
    V = convert(V, To.Element);
    std::vector<uint32_t> Elements(To.Count, V.Id);
    return {composite(T, Elements), T, 7, AST};
  }
  if (From.K == SpirvType::Vector && To.K != SpirvType::Vector) {
    V.Id = valueOp(OpCompositeExtract, From.Element, {V.Id, 0});
    V.Type = From.Element; return convert(V, T, AST);
  }
  if (From.K == SpirvType::Vector && To.K == SpirvType::Vector && From.Count != To.Count) {
    std::vector<uint32_t> Elements;
    for (uint32_t I = 0; I < To.Count; ++I) {
      if (I >= From.Count) Elements.push_back(zero(To.Element));
      else Elements.push_back(convert({valueOp(OpCompositeExtract, From.Element, {V.Id, I}), From.Element}, To.Element).Id);
    }
    return {composite(T, Elements), T, 7, AST};
  }
  SpirvType F = From.K == SpirvType::Vector ? Types[From.Element] : From;
  SpirvType D = To.K == SpirvType::Vector ? Types[To.Element] : To;
  uint32_t Op = 0;
  if (D.K == SpirvType::Bool)
    return {valueOp(F.K == SpirvType::Float ? OpFUnordNotEqual : OpINotEqual,
                     T, {V.Id, zero(V.Type)}), T, 7, AST};
  if (F.K == SpirvType::Bool) {
    uint32_t One = constant(To.K == SpirvType::Vector ? To.Element : T,
        D.K == SpirvType::Float ? (D.Bits == 64 ? 0x3ff0000000000000ull : D.Bits == 16 ? 0x3c00u : 0x3f800000u) : 1u);
    if (To.K == SpirvType::Vector) One = composite(T, std::vector<uint32_t>(To.Count, One), true);
    return {valueOp(OpSelect, T, {V.Id, One, zero(T)}), T, 7, AST};
  }
  if (F.K == SpirvType::Float && D.K == SpirvType::Float) Op = OpFConvert;
  else if (F.K == SpirvType::Int && D.K == SpirvType::Float) Op = F.Signed ? OpConvertSToF : OpConvertUToF;
  else if (F.K == SpirvType::Float && D.K == SpirvType::Int) Op = D.Signed ? OpConvertFToS : OpConvertFToU;
  else if (F.K == SpirvType::Int && D.K == SpirvType::Int)
    Op = F.Bits == D.Bits ? OpBitcast : (D.Signed ? OpSConvert : OpUConvert);
  if (Op) return {valueOp(Op, T, {V.Id}), T, 7, AST};
  if (F.K == D.K && (F.K == SpirvType::Struct || F.K == SpirvType::Array || F.K == SpirvType::Matrix))
    return {repack(V.Id, V.Type, T), T, 7, AST};
  fail("unsupported shader value conversion"); return {};
}
}
