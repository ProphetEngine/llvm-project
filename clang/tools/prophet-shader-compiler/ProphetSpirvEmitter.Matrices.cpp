//===-- ProphetSpirvEmitter.Matrices.cpp - Logical matrix operations --------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

uint32_t ProphetSpirvEmitter::matrixScalarType(uint32_t T) {
  uint32_t Column = Types[T].Element;
  return Types[Column].K == SpirvType::Vector ? Types[Column].Element : Column;
}
SpirvValue ProphetSpirvEmitter::matrixAddress(SpirvValue Base, uint32_t Row,
                                             uint32_t Column, QualType Scalar) {
  uint32_t Element = matrixScalarType(Base.Type);
  const auto Shape = Types[Base.Type];
  if (!Shape.NativeMatrix && Shape.RowMajor) {
    if (Shape.Count == 1) return access(Base, {Row}, Scalar, Element);
    return access(Base, {Row, Column}, Scalar, Element);
  }
  if (Shape.Rows == 1) return access(Base, {Column}, Scalar, Element);
  return access(Base, {Column, Row}, Scalar, Element);
}
SpirvValue ProphetSpirvEmitter::matrixElement(SpirvValue Base, uint32_t Row,
                                             uint32_t Column) {
  Base = load(Base);
  uint32_t Scalar = matrixScalarType(Base.Type);
  const auto Shape = Types[Base.Type];
  bool RowMajor = !Shape.NativeMatrix && Shape.RowMajor;
  std::vector<uint32_t> Args{Base.Id, RowMajor ? Row : Column};
  if ((RowMajor ? Shape.Count : Shape.Rows) > 1) Args.push_back(RowMajor ? Column : Row);
  return {valueOp(OpCompositeExtract, Scalar, Args), Scalar};
}
SpirvValue ProphetSpirvEmitter::matrixTranspose(SpirvValue Value, QualType Result) {
  Value = load(Value); uint32_t T = type(Result);
  if (Types[Value.Type].NativeMatrix && Types[T].NativeMatrix)
    return {valueOp(OpTranspose, T, {Value.Id}), T, 7, Result};
  std::vector<uint32_t> Columns;
  for (uint32_t C = 0; C < Types[T].Count; ++C) {
    std::vector<uint32_t> Rows;
    for (uint32_t R = 0; R < Types[T].Rows; ++R)
      Rows.push_back(matrixElement(Value, C, R).Id);
    Columns.push_back(composite(Types[T].Element, Rows));
  }
  return {composite(T, Columns), T, 7, Result};
}
SpirvValue ProphetSpirvEmitter::matrixMultiply(SpirvValue A, SpirvValue B,
                                              QualType Result) {
  A = load(A); B = load(B); uint32_t T = type(Result);
  const auto AT = Types[A.Type], BT = Types[B.Type], RT = Types[T];
  bool AM = AT.K == SpirvType::Matrix, BM = BT.K == SpirvType::Matrix;
  bool AV = AT.K == SpirvType::Vector, BV = BT.K == SpirvType::Vector;
  if ((!AM && !AV) || (!BM && !BV))
    return arithmetic(BO_Mul, A, B, Result, {});
  uint32_t Scalar = RT.K == SpirvType::Matrix ? matrixScalarType(T) :
      RT.K == SpirvType::Vector ? RT.Element : T;
  if (Types[Scalar].K == SpirvType::Float &&
      (!AM || AT.NativeMatrix) && (!BM || BT.NativeMatrix)) {
    uint32_t Op = AM && BM ? OpMatrixTimesMatrix : AM ? OpMatrixTimesVector :
                  BM ? OpVectorTimesMatrix : OpDot;
    return {valueOp(Op, T, {A.Id, B.Id}), T, 7, Result};
  }
  QualType ScalarAST = Result;
  if (const auto *M = Result->getAs<ConstantMatrixType>()) ScalarAST = M->getElementType();
  else if (const auto *V = Result->getAs<VectorType>()) ScalarAST = V->getElementType();
  auto Lane = [&](SpirvValue V, uint32_t I) -> SpirvValue {
    return {valueOp(OpCompositeExtract, Types[V.Type].Element, {V.Id, I}), Types[V.Type].Element};
  };
  uint32_t Inner = AT.Count;
  auto Element = [&](uint32_t Row, uint32_t Column) -> uint32_t {
    SpirvValue Sum{zero(Scalar), Scalar, 7, ScalarAST};
    for (uint32_t K = 0; K < Inner; ++K) {
      SpirvValue Left = convert(AM ? matrixElement(A, Row, K) : Lane(A, K), Scalar);
      SpirvValue Right = convert(BM ? matrixElement(B, K, Column) : Lane(B, K), Scalar);
      Sum = arithmetic(BO_Add, Sum, arithmetic(BO_Mul, Left, Right, ScalarAST, {}), ScalarAST, {});
    }
    return Sum.Id;
  };
  if (RT.K == SpirvType::Matrix) {
    std::vector<uint32_t> Columns;
    for (uint32_t C = 0; C < RT.Count; ++C) {
      std::vector<uint32_t> Rows;
      for (uint32_t R = 0; R < RT.Rows; ++R) Rows.push_back(Element(R, C));
      Columns.push_back(composite(RT.Element, Rows));
    }
    return {composite(T, Columns), T, 7, Result};
  }
  if (RT.K == SpirvType::Vector) {
    std::vector<uint32_t> Elements;
    for (uint32_t I = 0; I < RT.Count; ++I) Elements.push_back(Element(AM ? I : 0, AM ? 0 : I));
    return {composite(T, Elements), T, 7, Result};
  }
  return {Element(0, 0), T, 7, Result};
}
} // namespace prophet
