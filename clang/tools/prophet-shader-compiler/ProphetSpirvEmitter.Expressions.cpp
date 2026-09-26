//===-- ProphetSpirvEmitter.Expressions.cpp - Common AST expressions ------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"
#include "llvm/ADT/SmallVector.h"
#include "clang/Basic/SourceManager.h"

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

SpirvValue ProphetSpirvEmitter::initialize(QualType T, ArrayRef<const Expr *> Values) {
  uint32_t ST = type(T); SpirvType Info = Types[ST];
  if (Values.empty()) return {zero(ST), ST, 7, T};
  if (Info.K != SpirvType::Struct && Info.K != SpirvType::Array &&
      Info.K != SpirvType::Vector && Info.K != SpirvType::Matrix)
    return convert(expression(Values.front()), ST, T);
  std::vector<SpirvValue> Parts;
  for (const Expr *E : Values) {
    SpirvValue V = load(expression(E));
    if ((Info.K == SpirvType::Vector || Info.K == SpirvType::Matrix) && Types[V.Type].K == SpirvType::Vector) {
      for (uint32_t I = 0; I < Types[V.Type].Count; ++I) {
        uint32_t Element = Types[V.Type].Element;
        Parts.push_back({valueOp(OpCompositeExtract, Element, {V.Id, I}), Element});
      }
    } else Parts.push_back(V);
  }
  if (Info.K == SpirvType::Matrix) {
    uint32_t Scalar = matrixScalarType(ST);
    std::vector<uint32_t> Columns;
    for (uint32_t C = 0; C < Info.Count; ++C) {
      std::vector<uint32_t> Rows;
      for (uint32_t R = 0; R < Info.Rows; ++R) {
        uint32_t Index = R * Info.Count + C;
        Rows.push_back(Parts.size() == 1 ? convert(Parts[0], Scalar).Id :
            Index < Parts.size() ? convert(Parts[Index], Scalar).Id : zero(Scalar));
      }
      Columns.push_back(composite(Info.Element, Rows));
    }
    return {composite(ST, Columns), ST, 7, T};
  }
  uint32_t Count = Info.K == SpirvType::Struct ? Info.Members.size() : Info.Count;
  std::vector<uint32_t> IDs;
  for (uint32_t I = 0; I < Count; ++I) {
    uint32_t Element = Info.K == SpirvType::Struct ? Info.Members[I] : Info.Element;
    if (I < Parts.size()) IDs.push_back(convert(Parts[I], Element).Id);
    else if (Info.K == SpirvType::Vector && Parts.size() == 1)
      IDs.push_back(convert(Parts[0], Element).Id);
    else IDs.push_back(zero(Element));
  }
  return {composite(ST, IDs), ST, 7, T};
}
SpirvValue ProphetSpirvEmitter::expression(const Expr *E) {
  if (!E || !Diagnostic.empty()) return {};
  if (const auto *P = dyn_cast<ParenExpr>(E)) return expression(P->getSubExpr());
  if (const auto *C = dyn_cast<ExprWithCleanups>(E)) return expression(C->getSubExpr());
  if (const auto *M = dyn_cast<MaterializeTemporaryExpr>(E)) {
    SpirvValue V = expression(M->getSubExpr());
    if (V.Address || V.Resource || V.Closure) return V;
    SpirvValue Temporary = temporary(M->getType()); store(Temporary, V); return Temporary;
  }
  if (const auto *B = dyn_cast<CXXBindTemporaryExpr>(E)) {
    if (!B->getTemporary()->getDestructor()->isTrivial()) {
      fail("nontrivial shader temporary destruction is not supported", E->getExprLoc()); return {};
    }
    return expression(B->getSubExpr());
  }
  if (const auto *D = dyn_cast<CXXDefaultArgExpr>(E)) return expression(D->getExpr());
  if (const auto *D = dyn_cast<CXXDefaultInitExpr>(E)) return expression(D->getExpr());
  if (const auto *O = dyn_cast<OpaqueValueExpr>(E)) {
    if (auto I = OpaqueValues.find(O); I != OpaqueValues.end()) return I->second;
    return expression(O->getSourceExpr());
  }
  if (const auto *O = dyn_cast<HLSLOutArgExpr>(E)) return expression(O->getArgLValue());
  if (const auto *R = dyn_cast<DeclRefExpr>(E)) {
    if (const auto *Binding = dyn_cast<BindingDecl>(R->getDecl()))
      return expression(Binding->getBinding());
    const ValueDecl *D = cast<ValueDecl>(R->getDecl()->getCanonicalDecl());
    for (auto I = Calls.rbegin(); I != Calls.rend(); ++I)
      if (I->Closure)
        if (auto C = I->Closure->Captures.find(D); C != I->Closure->Captures.end()) return C->second;
    if (auto I = Locals.find(D); I != Locals.end()) return I->second;
    if (auto I = Globals.find(D); I != Globals.end()) {
      if (I->second.Id) UsedGlobals.insert(I->second.Id);
      return I->second;
    }
    if (const auto *V = dyn_cast<VarDecl>(D); V && V->hasInit()) {
      SpirvValue Constant = constantExpr(V->getInit());
      if (Constant) return Constant;
    }
    if (const auto *V = dyn_cast<EnumConstantDecl>(D))
      return {constant(type(E->getType()), V->getInitVal().getLimitedValue()), type(E->getType()), 7, E->getType()};
    fail("shader variable is not bound: " + R->getDecl()->getNameAsString(), E->getExprLoc()); return {};
  }
  if (const auto *L = dyn_cast<LambdaExpr>(E)) {
    SpirvValue V; V.ASTType = L->getType(); V.Closure = std::make_shared<SpirvClosure>();
    V.Closure->Record = L->getLambdaClass();
    auto Init = L->capture_init_begin();
    for (const auto &Capture : L->captures()) {
      const Expr *Source = *Init++;
      if (!Capture.capturesVariable()) {
        fail("shader lambda capture must name a shader value", E->getExprLoc()); return {};
      }
      SpirvValue Captured = expression(Source);
      if (Capture.getCaptureKind() != LCK_ByRef && !Captured.Resource && !Captured.Closure) {
        SpirvValue Snapshot = temporary(Capture.getCapturedVar()->getType());
        store(Snapshot, Captured); Captured = Snapshot;
      }
      auto *Key = cast<ValueDecl>(Capture.getCapturedVar()->getCanonicalDecl());
      V.Closure->Captures[Key] = Captured;
      if (Capture.getCaptureKind() == LCK_ByRef) V.Closure->Borrowed.insert(Key);
    }
    return V;
  }
  if (const auto *T = dyn_cast<CXXThisExpr>(E)) {
    if (Calls.empty()) { fail("shader this is not bound", T->getExprLoc()); return {}; }
    return Calls.back().This;
  }
  // Constant evaluation is performed by Clang, including consteval recursion
  // and instantiated constexpr templates. Runtime calls use the same lowering.
  if (!isa<CallExpr>(E) || !E->HasSideEffects(Ctx)) {
    if (!E->getType()->isRecordType())
      if (SpirvValue V = constantExpr(E)) return V;
  }
  if (const auto *C = dyn_cast<CastExpr>(E)) {
    SpirvValue V = expression(C->getSubExpr());
    if (V.Resource && V.Resource->Uniform && C->getCastKind() == CK_LValueToRValue &&
        !C->getType()->isHLSLResourceRecord()) {
      auto *R = V.Resource;
      return access(V, {constant(integerType(), 0)}, R->ElementType, R->Element);
    }
    if (C->getCastKind() == CK_DerivedToBase || C->getCastKind() == CK_UncheckedDerivedToBase) {
      QualType Current = C->getSubExpr()->getType().getNonReferenceType();
      if (Current->isPointerType()) Current = Current->getPointeeType();
      for (const CXXBaseSpecifier *Base : C->path()) {
        auto *Record = Current->getAsCXXRecordDecl(); unsigned Index = 0;
        for (const auto &Candidate : Record->bases()) {
          if (Ctx.hasSameUnqualifiedType(Candidate.getType(), Base->getType())) break;
          ++Index;
        }
        V = access(V, {constant(integerType(), Index)}, Base->getType()); Current = Base->getType();
      }
      return V;
    }
    if (C->getType()->isPointerType() || V.Resource || V.Closure || V.Stream) return V;
    if (C->getCastKind() == CK_LValueToRValue) return convert(V, type(C->getType()), C->getType());
    if (C->getCastKind() == CK_NoOp ||
        (C->getCastKind() == CK_AddressSpaceConversion && C->isGLValue()) ||
        C->getCastKind() == CK_ArrayToPointerDecay ||
        C->getType()->isReferenceType() || C->getType()->isPointerType() || V.Resource || V.Closure)
      return V;
    if (C->getCastKind() == CK_ToVoid) return {};
    if (C->getCastKind() == CK_BitCast || C->getCastKind() == CK_LValueBitCast) {
      V = load(V); uint32_t T = type(C->getType());
      return {valueOp(OpBitcast, T, {V.Id}), T, 7, C->getType()};
    }
    return convert(V, type(C->getType()), C->getType());
  }
  if (const auto *P = dyn_cast<CXXParenListInitExpr>(E)) {
    std::vector<const Expr *> Values;
    for (const Expr *Part : P->getInitExprs()) Values.push_back(Part);
    return initialize(P->getType(), Values);
  }
  if (const auto *I = dyn_cast<InitListExpr>(E)) {
    if (I->isSyntacticForm() && I->getSemanticForm()) I = I->getSemanticForm();
    std::vector<const Expr *> Values;
    for (const Expr *Part : I->inits()) Values.push_back(Part);
    return initialize(I->getType(), Values);
  }
  if (isa<ImplicitValueInitExpr>(E) || isa<CXXScalarValueInitExpr>(E))
    return {zero(type(E->getType())), type(E->getType()), 7, E->getType()};
  if (const auto *C = dyn_cast<CXXConstructExpr>(E)) {
    if (C->getConstructor()->isCopyOrMoveConstructor() && C->getNumArgs() == 1 &&
        (C->getConstructor()->isTrivial() || C->getConstructor()->getParent()->isLambda() ||
         C->getConstructor()->getParent()->isHLSLBuiltinRecord()))
      return copyClosure(load(expression(C->getArg(0))));
    if (C->getConstructor()->isTrivial() && !C->getNumArgs())
      return {zero(type(E->getType())), type(E->getType()), 7, E->getType()};
    SpirvValue Object = temporary(E->getType());
    std::function<void(SpirvValue, QualType)> Construct = [&](SpirvValue Target, QualType T) {
      if (const auto *A = Ctx.getAsConstantArrayType(T)) {
        for (uint64_t I = 0; I < A->getSize().getLimitedValue(); ++I)
          Construct(access(Target, {constant(integerType(), I)}, A->getElementType()), A->getElementType());
        return;
      }
      std::vector<SpirvValue> Args;
      for (const Expr *Arg : C->arguments()) Args.push_back(expression(Arg));
      callFunction(C->getConstructor(), Args, Target);
    };
    Construct(Object, E->getType()); return Object;
  }
  if (const auto *M = dyn_cast<MemberExpr>(E)) {
    SpirvValue Base = expression(M->getBase());
    const auto *Field = dyn_cast<FieldDecl>(M->getMemberDecl());
    if (!Field) { fail("shader member is not a field", E->getExprLoc()); return {}; }
    uint32_t Index = Field->getFieldIndex();
    if (const auto *C = dyn_cast<CXXRecordDecl>(Field->getParent())) Index += C->getNumBases();
    if (Base.Resource && Base.Resource->Uniform) {
      Base = access(Base, {constant(integerType(), 0)}, Base.Resource->ElementType, Base.Resource->Element);
    }
    if (Base.Address) {
      uint32_t MemberType = Types[Base.Type].K == SpirvType::Struct && Index < Types[Base.Type].Members.size()
          ? Types[Base.Type].Members[Index] : type(Field->getType());
      return access(Base, {constant(integerType(), Index)}, Field->getType(), MemberType);
    }
    return extract(Base, {Index}, Field->getType());
  }
  if (const auto *V = dyn_cast<ExtVectorElementExpr>(E)) {
    SpirvValue Base = expression(V->getBase());
    SmallVector<uint32_t, 4> Elements; V->getEncodedElementAccess(Elements);
    Base.Swizzle.assign(Elements.begin(), Elements.end()); Base.ASTType = E->getType();
    return Base;
  }
  if (const auto *M = dyn_cast<MatrixSingleSubscriptExpr>(E)) {
    SpirvValue Base = expression(M->getBase());
    uint32_t Row = load(expression(M->getRowIdx())).Id;
    for (uint32_t Column = 0; Column < Types[Base.Type].Count; ++Column)
      Base.MatrixComponents.emplace_back(Row, constant(integerType(), Column));
    Base.ASTType = E->getType(); return Base;
  }
  if (const auto *M = dyn_cast<MatrixElementExpr>(E)) {
    SpirvValue Base = expression(M->getBase());
    SmallVector<uint32_t, 4> Indices; M->getEncodedElementAccess(Indices);
    uint32_t Columns = Types[Base.Type].Count;
    for (uint32_t Index : Indices)
      Base.MatrixComponents.emplace_back(constant(integerType(), Index / Columns),
                                          constant(integerType(), Index % Columns));
    Base.ASTType = E->getType(); return Base;
  }
  if (const auto *M = dyn_cast<MatrixSubscriptExpr>(E)) {
    if (M->isIncomplete()) { fail("matrix subscript requires row and column", E->getExprLoc()); return {}; }
    SpirvValue Base = expression(M->getBase());
    if (!Base.Address) {
      SpirvValue Temporary = temporary(M->getBase()->getType()); store(Temporary, Base); Base = Temporary;
    }
    uint32_t Row = load(expression(M->getRowIdx())).Id;
    uint32_t Column = load(expression(M->getColumnIdx())).Id;
    return matrixAddress(Base, Row, Column, E->getType());
  }
  if (const auto *A = dyn_cast<ArraySubscriptExpr>(E)) {
    SpirvValue Base = expression(A->getBase()), Index = load(expression(A->getIdx()));
    if (Base.Resource) return resourceIndex(Base, Index, E->getType());
    if (const auto *Vector = A->getBase()->getType()->getAs<VectorType>();
        Vector && Vector->getNumElements() == 1) return Base;
    if (Base.Address) {
      uint32_t Element = Types[Base.Type].Element;
      return access(Base, {Index.Id}, E->getType(), Element);
    }
    Base = load(Base);
    if (Types[Base.Type].K == SpirvType::Vector)
      return {valueOp(OpVectorExtractDynamic, type(E->getType()), {Base.Id, Index.Id}), type(E->getType()), 7, E->getType()};
    SpirvValue Copy = temporary(Base.ASTType); store(Copy, Base);
    return access(Copy, {Index.Id}, E->getType());
  }
  if (const auto *B = dyn_cast<BinaryOperator>(E)) return binary(B);
  if (const auto *U = dyn_cast<UnaryOperator>(E)) return unary(U);
  if (const auto *C = dyn_cast<ConditionalOperator>(E)) {
    SpirvValue Result = temporary(E->getType());
    SpirvValue Condition = convert(expression(C->getCond()), boolType());
    uint32_t Yes = id(), No = id(), Merge = id();
    Body.emit(OpSelectionMerge, {Merge, 0}); conditionalBranch(Condition.Id, Yes, No);
    ControlConditions.push_back(Condition.Id);
    label(Yes); store(Result, expression(C->getTrueExpr())); branch(Merge);
    label(No); store(Result, expression(C->getFalseExpr())); branch(Merge);
    ControlConditions.pop_back(); label(Merge); return load(Result);
  }
  if (const auto *C = dyn_cast<CallExpr>(E)) return call(C);
  if (const auto *L = dyn_cast<ArrayInitLoopExpr>(E)) {
    const auto *Common = L->getCommonExpr();
    auto Saved = OpaqueValues;
    OpaqueValues[Common] = expression(Common->getSourceExpr());
    uint32_t T = type(E->getType()); std::vector<uint32_t> Values;
    for (uint32_t I = 0; I < Types[T].Count; ++I) {
      ArrayInitIndices.push_back(I); Values.push_back(load(expression(L->getSubExpr())).Id);
      ArrayInitIndices.pop_back();
    }
    OpaqueValues = std::move(Saved); return {composite(T, Values), T, 7, E->getType()};
  }
  if (isa<ArrayInitIndexExpr>(E)) {
    if (ArrayInitIndices.empty()) { fail("array initializer index is unbound", E->getExprLoc()); return {}; }
    uint32_t T = type(E->getType()); return {constant(T, ArrayInitIndices.back()), T, 7, E->getType()};
  }
  if (const auto *P = dyn_cast<PackExpansionExpr>(E)) return expression(P->getPattern());
  if (const auto *S = dyn_cast<SubstNonTypeTemplateParmExpr>(E)) return expression(S->getReplacement());
  if (const auto *C = dyn_cast<ConstantExpr>(E)) return expression(C->getSubExpr());
  fail(("unsupported runtime shader expression: " + Twine(E->getStmtClassName())).str(), E->getExprLoc());
  return {};
}

SpirvValue ProphetSpirvEmitter::arithmetic(BinaryOperatorKind Op, SpirvValue A,
    SpirvValue B, QualType Result, SourceLocation Loc) {
  A = load(A); B = load(B);
  if (!A || !B) return {};
  uint32_t T = type(Result), Operand = A.Type;
  if (const auto *M = Result->getAs<ConstantMatrixType>()) {
    if (Op == BO_Mul && !Ctx.getLangOpts().HLSL) {
      if (Types[A.Type].K == SpirvType::Matrix && Types[B.Type].K == SpirvType::Matrix)
        return {valueOp(OpMatrixTimesMatrix, T, {A.Id, B.Id}), T, 7, Result};
      if (Types[B.Type].K == SpirvType::Matrix) std::swap(A, B);
      return {valueOp(OpMatrixTimesScalar, T, {A.Id, B.Id}), T, 7, Result};
    }
    A = convert(A, T); B = convert(B, T);
    uint32_t ColumnType = Types[T].Element;
    QualType ColumnAST = Ctx.getExtVectorType(M->getElementType(), M->getNumRows());
    std::vector<uint32_t> Columns;
    for (uint32_t C = 0; C < M->getNumColumns(); ++C) {
      SpirvValue Left{valueOp(OpCompositeExtract, ColumnType, {A.Id, C}), ColumnType, 7, ColumnAST};
      SpirvValue Right{valueOp(OpCompositeExtract, ColumnType, {B.Id, C}), ColumnType, 7, ColumnAST};
      Columns.push_back(arithmetic(Op, Left, Right, ColumnAST, Loc).Id);
    }
    return {composite(T, Columns), T, 7, Result};
  }
  if (Types[B.Type].K == SpirvType::Vector && Types[A.Type].K != SpirvType::Vector)
    Operand = B.Type;
  A = convert(A, Operand); B = convert(B, Operand);
  SpirvType Element = Types[Operand];
  if (Element.K == SpirvType::Vector) Element = Types[Element.Element];
  bool Floating = Element.K == SpirvType::Float, Signed = Element.Signed;
  uint32_t Code = 0;
  switch (Op) {
  case BO_Add: Code = Floating ? OpFAdd : OpIAdd; break;
  case BO_Sub: Code = Floating ? OpFSub : OpISub; break;
  case BO_Mul: Code = Floating ? OpFMul : OpIMul; break;
  case BO_Div: Code = Floating ? OpFDiv : Signed ? OpSDiv : OpUDiv; break;
  case BO_Rem: Code = Floating ? OpFRem : Signed ? OpSRem : OpUMod; break;
  case BO_Shl:
  case BO_Shr: {
    uint32_t Mask = constant(integerType(Element.Bits, Signed), Element.Bits - 1);
    if (Types[Operand].K == SpirvType::Vector)
      Mask = composite(Operand, std::vector<uint32_t>(Types[Operand].Count, Mask), true);
    B.Id = valueOp(OpBitwiseAnd, Operand, {B.Id, Mask});
    Code = Op == BO_Shl ? OpShiftLeftLogical : Signed ? OpShiftRightArithmetic : OpShiftRightLogical;
    break;
  }
  case BO_And: Code = Element.K == SpirvType::Bool ? OpLogicalAnd : OpBitwiseAnd; break;
  case BO_Or: Code = Element.K == SpirvType::Bool ? OpLogicalOr : OpBitwiseOr; break;
  case BO_Xor: Code = Element.K == SpirvType::Bool ? OpLogicalNotEqual : OpBitwiseXor; break;
  case BO_EQ: Code = Floating ? OpFOrdEqual : Element.K == SpirvType::Bool ? OpLogicalEqual : OpIEqual; break;
  case BO_NE: Code = Floating ? OpFUnordNotEqual : Element.K == SpirvType::Bool ? OpLogicalNotEqual : OpINotEqual; break;
  case BO_LT: Code = Floating ? OpFOrdLessThan : Signed ? OpSLessThan : OpULessThan; break;
  case BO_LE: Code = Floating ? OpFOrdLessThanEqual : Signed ? OpSLessThanEqual : OpULessThanEqual; break;
  case BO_GT: Code = Floating ? OpFOrdGreaterThan : Signed ? OpSGreaterThan : OpUGreaterThan; break;
  case BO_GE: Code = Floating ? OpFOrdGreaterThanEqual : Signed ? OpSGreaterThanEqual : OpUGreaterThanEqual; break;
  case BO_LAnd: Code = OpLogicalAnd; break;
  case BO_LOr: Code = OpLogicalOr; break;
  default: fail("unsupported shader arithmetic operation", Loc); return {};
  }
  if (Code == OpSRem) {
    uint32_t Quotient = valueOp(OpSDiv, Operand, {A.Id, B.Id});
    uint32_t Product = valueOp(OpIMul, Operand, {Quotient, B.Id});
    return {valueOp(OpISub, T, {A.Id, Product}), T, 7, Result};
  }
  return {valueOp(Code, T, {A.Id, B.Id}), T, 7, Result};
}
SpirvValue ProphetSpirvEmitter::binary(const BinaryOperator *E) {
  BinaryOperatorKind Op = E->getOpcode();
  if (Op == BO_Comma) { expression(E->getLHS()); return expression(E->getRHS()); }
  if (Op == BO_Assign) {
    SpirvValue Target = expression(E->getLHS());
    SpirvValue Value = expression(E->getRHS());
    // Mesh outputs are write-only. Preserve the assignment's value for chains
    // instead of introducing a load from the output just written.
    if (CurrentStage == 8 && Target.Storage == StorageClass::Output) {
      Value = convert(Value, type(E->getType()), E->getType());
      if (!store(Target, Value)) return {};
      return Value;
    }
    store(Target, Value); return Target;
  }
  if (E->isCompoundAssignmentOp()) {
    SpirvValue Target = expression(E->getLHS());
    SpirvValue Value = arithmetic(BinaryOperator::getOpForCompoundAssignment(Op),
        Target, expression(E->getRHS()), E->getType(), E->getExprLoc());
    store(Target, Value); return Target;
  }
  if ((Op == BO_LAnd || Op == BO_LOr) && !E->getType()->isVectorType()) {
    SpirvValue Result = temporary(E->getType());
    bool IsAnd = Op == BO_LAnd;
    store(Result, {constantBool(!IsAnd), boolType()});
    SpirvValue Left = convert(expression(E->getLHS()), boolType());
    uint32_t Right = id(), Merge = id(); Body.emit(OpSelectionMerge, {Merge, 0});
    conditionalBranch(Left.Id, IsAnd ? Right : Merge, IsAnd ? Merge : Right);
    ControlConditions.push_back(Left.Id);
    label(Right); store(Result, convert(expression(E->getRHS()), boolType())); branch(Merge);
    ControlConditions.pop_back(); label(Merge); return load(Result);
  }
  return arithmetic(Op, expression(E->getLHS()), expression(E->getRHS()), E->getType(), E->getExprLoc());
}
SpirvValue ProphetSpirvEmitter::unary(const UnaryOperator *E) {
  SpirvValue V = expression(E->getSubExpr());
  if (E->getOpcode() == UO_AddrOf || E->getOpcode() == UO_Deref) return V;
  if (E->isIncrementDecrementOp()) {
    SpirvValue Previous = load(V); uint32_t T = Previous.Type;
    SpirvType Info = Types[T]; if (Info.K == SpirvType::Vector) Info = Types[Info.Element];
    uint64_t One = Info.K == SpirvType::Float
        ? (Info.Bits == 64 ? 0x3ff0000000000000ull : Info.Bits == 16 ? 0x3c00u : 0x3f800000u) : 1;
    uint32_t Scalar = Types[T].K == SpirvType::Vector ? Types[T].Element : T;
    SpirvValue Increment = convert({constant(Scalar, One), Scalar}, T);
    SpirvValue Updated = arithmetic(E->isIncrementOp() ? BO_Add : BO_Sub,
        Previous, Increment, E->getType(), E->getExprLoc());
    store(V, Updated); return E->isPostfix() ? Previous : Updated;
  }
  V = load(V); uint32_t T = type(E->getType());
  if (E->getOpcode() == UO_Plus) return V;
  if (E->getOpcode() == UO_LNot) {
    V = convert(V, Types[T].K == SpirvType::Vector ? vectorType(boolType(), Types[T].Count) : boolType());
    return {valueOp(OpLogicalNot, T, {V.Id}), T, 7, E->getType()};
  }
  if (E->getOpcode() == UO_Not) return {valueOp(OpNot, T, {V.Id}), T, 7, E->getType()};
  if (E->getOpcode() == UO_Minus) {
    SpirvType Info = Types[T]; if (Info.K == SpirvType::Vector) Info = Types[Info.Element];
    return {valueOp(Info.K == SpirvType::Float ? OpFNegate : OpSNegate, T, {V.Id}), T, 7, E->getType()};
  }
  fail("unsupported shader unary operation", E->getExprLoc()); return {};
}
SpirvValue ProphetSpirvEmitter::copyClosure(SpirvValue V) {
  if (!V.Closure) return V;
  auto Copy = std::make_shared<SpirvClosure>(*V.Closure);
  for (auto &[D, Captured] : Copy->Captures) {
    if (Copy->Borrowed.contains(D)) continue;
    if (Captured.Closure) Captured = copyClosure(Captured);
    else if (!Captured.Resource) {
      SpirvValue Snapshot = temporary(Captured.ASTType);
      store(Snapshot, Captured); Captured = Snapshot;
    }
  }
  V.Closure = std::move(Copy); return V;
}
SpirvValue ProphetSpirvEmitter::call(const CallExpr *E) {
  if (SpirvValue Constant = constantExpr(E)) return Constant;
  const FunctionDecl *F = E->getDirectCallee();
  if (!F) { fail("shader indirect calls require a statically resolved target", E->getExprLoc()); return {}; }
  SpirvValue Object;
  unsigned First = 0;
  if (const auto *M = dyn_cast<CXXMemberCallExpr>(E)) Object = expression(M->getImplicitObjectArgument());
  else if (const auto *O = dyn_cast<CXXOperatorCallExpr>(E)) {
    if (const auto *M = dyn_cast<CXXMethodDecl>(F); M && !M->isStatic()) {
      Object = expression(O->getArg(0)); First = 1;
    }
  }
  std::vector<SpirvValue> Args;
  for (unsigned I = First; I < E->getNumArgs(); ++I) {
    const Expr *Argument = E->getArg(I);
    if (F->getNameAsString() == "DispatchMesh" && I == 3)
      Argument = Argument->IgnoreParenImpCasts();
    SpirvValue Value = expression(Argument);
    if (I - First < F->getNumParams() &&
        !F->getParamDecl(I - First)->hasAttr<HLSLParamModifierAttr>() &&
        F->getParamDecl(I - First)->getType()->isReferenceType() &&
        Argument->IgnoreParenImpCasts()->isGLValue() && !Value.Address &&
        !Value.Interface && !Value.Resource && !Value.Closure && !Value.Stream) {
      fail("runtime-selected reference requires a statically known shader address", Argument->getExprLoc()); return {};
    }
    Args.push_back(Value);
  }
  if (const auto *Method = dyn_cast<CXXMethodDecl>(F))
    if (Method->isTrivial() && (Method->isCopyAssignmentOperator() || Method->isMoveAssignmentOperator())) {
      if (Args.size() != 1) { fail("invalid copy assignment", E->getExprLoc()); return {}; }
      store(Object, Args[0]); return Object;
    }
  std::string Name = F->getNameAsString();
  bool Handled = false;
  if (Object.Resource) {
    SpirvValue R = resourceCall(Name, Object, E, Args, Handled);
    if (Handled) return R;
  }
  if (const auto *M = dyn_cast<CXXMethodDecl>(F)) {
    StringRef Class = M->getParent()->getName();
    if (Object.Stream) {
      if (Name == "Append" && Args.size() == 1) {
        interfaceOutput(nullptr, load(Args[0])); Body.emit(OpEmitVertex); return {};
      }
      if (Name == "RestartStrip") { Body.emit(OpEndPrimitive); return {}; }
    }
    if ((Class == "InputPatch" || Class == "OutputPatch") && Name == "operator[]" && Args.size() == 1)
      return access(Object, {load(Args[0]).Id}, E->getType());
  }
  if (!F->getDefinition() || F->getBuiltinID() || Ctx.getSourceManager().isInSystemHeader(F->getLocation())) {
    SpirvValue R = intrinsic(Name, E, Args, Handled);
    if (Handled) return R;
  }
  return callFunction(F, Args, Object);
}
}
