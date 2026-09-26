//===-- ProphetSpirvEmitter.Images.cpp - Shared texture operations --------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

SpirvValue ProphetSpirvEmitter::resourceCall(StringRef Name, SpirvValue Object,
    const CallExpr *E, ArrayRef<SpirvValue> Args, bool &Handled) {
  Handled = true;
  SpirvResource &R = *Object.Resource;
  UsedGlobals.insert(R.Variable);
  if (Name == "operator[]" && Args.size() == 1)
    return resourceIndex(Object, Args[0], E->getType());
  if (R.K == SpirvResource::Buffer) {
    if (R.Uniform && Args.empty() &&
        (Name == "Load" || isa<CXXConversionDecl>(E->getDirectCallee())))
      return access(Object, {constant(integerType(), 0)}, R.ElementType, R.Element);
    if (Name.starts_with("Load") || Name.starts_with("Store")) {
      uint32_t Count = Name.empty() || !isDigit(Name.back()) ? 1 : Name.back() - '0';
      uint32_t Index = load(Args[0]).Id;
      QualType ResourceType = R.Decl->getType();
      while (const auto *A = Ctx.getAsArrayType(ResourceType)) ResourceType = A->getElementType();
      std::string Class = ResourceType->getAsCXXRecordDecl()->getNameAsString();
      if (StringRef(Class).contains("ByteAddress"))
        Index = valueOp(OpShiftRightLogical, integerType(), {Index, constant(integerType(), 2)});
      std::vector<uint32_t> Values;
      SpirvValue Input;
      if (Name.starts_with("Store")) Input = load(Args[1]);
      for (uint32_t I = 0; I < Count; ++I) {
        uint32_t Offset = I ? valueOp(OpIAdd, integerType(), {Index, constant(integerType(), I)}) : Index;
        SpirvValue Pointer = resourceIndex(Object, {Offset, integerType()}, R.ElementType);
        if (Name.starts_with("Load")) Values.push_back(load(Pointer).Id);
        else {
          uint32_t Part = Count == 1 ? Input.Id : valueOp(OpCompositeExtract, R.Element, {Input.Id, I});
          store(Pointer, {Part, R.Element});
        }
      }
      if (Name.starts_with("Store")) return {};
      uint32_t T = type(E->getType());
      uint32_t Packed = Count == 1 ? R.Element : vectorType(R.Element, Count);
      return convert({Count == 1 ? Values[0] : composite(Packed, Values), Packed}, T, E->getType());
    }
    if (Name == "GetDimensions") {
      uint32_t Member = R.CppRecord ? Types[R.Type].Members.size() - 1 : 0;
      uint32_t Length = valueOp(OpArrayLength, integerType(), {Object.Id, Member});
      QualType ResourceType = R.Decl->getType();
      while (const auto *A = Ctx.getAsArrayType(ResourceType)) ResourceType = A->getElementType();
      if (ResourceType->getAsCXXRecordDecl()->getName().contains("ByteAddress"))
        Length = valueOp(OpIMul, integerType(), {Length, constant(integerType(), 4)});
      if (!Args.empty()) store(Args[0], {Length, integerType()});
      if (Args.size() > 1) store(Args[1], {constant(integerType(), ((Types[R.Element].Size + Types[R.Element].Align - 1) / Types[R.Element].Align) * Types[R.Element].Align), integerType()});
      return {};
    }
    Handled = false; return {};
  }
  if (R.K == SpirvResource::Texture || R.K == SpirvResource::StorageTexture) {
    // The selected descriptor pointer belongs to this call, while the resource
    // declaration and image type remain shared by both source languages.
    SpirvResource Selected = R;
    if (Object.DescriptorSelected) Selected.Variable = Object.Id;
    return imageCall(Name, Selected, E, Args);
  }
  Handled = false; return {};
}
SpirvValue ProphetSpirvEmitter::imageCall(StringRef Name, SpirvResource &R,
    const CallExpr *E, ArrayRef<SpirvValue> Args) {
  uint32_t Image = valueOp(OpLoad, R.Type, {R.Variable});
  uint32_t T = E->getType()->isVoidType() ? 0 : type(E->getType());
  uint32_t CoordinateCount = R.Dimension == Dim::_1D ? 1 :
      R.Dimension == Dim::_3D || R.Dimension == Dim::Cube ? 3 : 2;
  CoordinateCount += R.Arrayed;
  auto Coordinate = [&](SpirvValue V, uint32_t Count, bool Integer) {
    V = load(V);
    uint32_t Scalar = Integer ? integerType(32, true) : floatType();
    uint32_t Destination = Count == 1 ? Scalar : vectorType(Scalar, Count);
    if (Types[V.Type].K == SpirvType::Struct) {
      std::vector<uint32_t> Values;
      for (uint32_t I = 0; I < Count; ++I) {
        uint32_t From = Types[V.Type].Members[I];
        Values.push_back(convert({valueOp(OpCompositeExtract, From, {V.Id, I}), From}, Scalar).Id);
      }
      return Count == 1 ? Values[0] : composite(Destination, Values);
    }
    return convert(V, Destination).Id;
  };
  auto ImageResult = [&](uint32_t Result, uint32_t ResultType) {
    return convert({Result, ResultType}, T, E->getType());
  };
  if (Name == "GetDimensions") {
    capability(Capability::ImageQuery);
    uint32_t Dimensions = R.Dimension == Dim::_1D ? 1 : R.Dimension == Dim::_3D ? 3 : 2;
    Dimensions += R.Arrayed;
    bool HasLod = R.ReadOnly && !R.Multisample && Args.size() > Dimensions;
    uint32_t QT = Dimensions == 1 ? integerType() : vectorType(integerType(), Dimensions);
    uint32_t Size;
    if (R.ReadOnly && !R.Multisample)
      Size = valueOp(OpImageQuerySizeLod, QT, {Image, HasLod ? load(Args[0]).Id : constant(integerType(), 0)});
    else Size = valueOp(OpImageQuerySize, QT, {Image});
    unsigned First = HasLod ? 1 : 0;
    for (uint32_t I = 0; I < Dimensions && First + I < Args.size(); ++I) {
      uint32_t Part = Dimensions == 1 ? Size : valueOp(OpCompositeExtract, integerType(), {Size, I});
      store(Args[First + I], {Part, integerType()});
    }
    if (First + Dimensions < Args.size()) {
      uint32_t Levels = valueOp(R.Multisample ? OpImageQuerySamples : OpImageQueryLevels, integerType(), {Image});
      store(Args[First + Dimensions], {Levels, integerType()});
    }
    return {};
  }
  uint32_t Color = vectorType(R.Element, 4);
  if (Name == "Load") {
    if (Args.empty()) { fail("texture Load requires a coordinate", E->getExprLoc()); return {}; }
    SpirvValue Location = load(Args[0]);
    uint32_t Coord = Coordinate(Location, CoordinateCount, true);
    uint32_t Op = R.ReadOnly ? OpImageFetch : OpImageRead;
    std::vector<uint32_t> Operands{Image, Coord};
    if (R.Multisample) {
      if (Args.size() < 2) { fail("multisample Load requires a sample index", E->getExprLoc()); return {}; }
      Operands.push_back(ImageOperands::Sample); Operands.push_back(load(Args[1]).Id);
    } else if (R.ReadOnly) {
      uint32_t Lod = constant(integerType(32, true), 0);
      if (Types[Location.Type].K == SpirvType::Vector && Types[Location.Type].Count > CoordinateCount)
        Lod = valueOp(OpCompositeExtract, Types[Location.Type].Element, {Location.Id, CoordinateCount});
      Operands.push_back(ImageOperands::Lod); Operands.push_back(Lod);
    }
    return ImageResult(valueOp(Op, Color, Operands), Color);
  }
  bool Gather = Name.starts_with("Gather"), Compare = Name.contains("Cmp");
  bool Sample = Name.starts_with("Sample");
  if (!Gather && !Sample) { fail("unsupported image operation: " + Name, E->getExprLoc()); return {}; }
  if (Args.size() < 2 || !Args[0].Resource) { fail("sampling requires a separate shader sampler", E->getExprLoc()); return {}; }
  if ((Name == "Sample" || Name == "SampleBias" || Name == "SampleCmp") && CurrentStage != 3 && !(ObjectMode && CurrentStage == 0)) {
    fail("implicit-derivative texture sampling requires a fragment entry", E->getExprLoc()); return {};
  }
  uint32_t CombinedType = internType(OpTypeSampledImage, {R.Type}, {});
  uint32_t Combined = valueOp(OpSampledImage, CombinedType, {Image, load(Args[0]).Id});
  uint32_t Coord = Coordinate(Args[1], CoordinateCount, false);
  std::vector<uint32_t> Operands{Combined, Coord};
  unsigned Next = 2;
  if (Compare) {
    if (Args.size() <= Next) { fail("comparison sampling requires a depth reference", E->getExprLoc()); return {}; }
    Operands.push_back(load(Args[Next++]).Id);
  } else if (Gather) {
    unsigned Component = Name.ends_with("Green") ? 1 : Name.ends_with("Blue") ? 2 : Name.ends_with("Alpha") ? 3 : 0;
    Operands.push_back(constant(integerType(), Component));
  }
  uint32_t Mask = 0;
  std::vector<uint32_t> ImageArguments;
  bool Explicit = Name == "SampleLevel" || Name == "SampleGrad" || Name == "SampleCmpLevelZero";
  if (Name == "SampleLevel" || Name == "SampleBias") {
    Mask |= Name == "SampleBias" ? ImageOperands::Bias : ImageOperands::Lod;
    if (Args.size() <= Next) { fail("sampling operation requires bias or LOD", E->getExprLoc()); return {}; }
    ImageArguments.push_back(load(Args[Next++]).Id);
  } else if (Name == "SampleCmpLevelZero") {
    Mask |= ImageOperands::Lod; ImageArguments.push_back(constant(floatType(), 0));
  } else if (Name == "SampleGrad") {
    if (Args.size() < Next + 2) { fail("SampleGrad requires two gradients", E->getExprLoc()); return {}; }
    Mask |= ImageOperands::Grad;
    uint32_t GradientCount = CoordinateCount - uint32_t(R.Arrayed);
    ImageArguments.push_back(Coordinate(Args[Next++], GradientCount, false));
    ImageArguments.push_back(Coordinate(Args[Next++], GradientCount, false));
  }
  if (Args.size() > Next) {
    uint32_t Offset = load(Args[Next]).Id;
    bool Constant = false;
    for (const auto &[K, V] : ConstantKeys) if (V == Offset) { Constant = true; break; }
    if (!Constant && !Gather) { fail("texture sampling offset must be constant", E->getExprLoc()); return {}; }
    Mask |= Constant ? ImageOperands::ConstOffset : ImageOperands::Offset;
    if (Gather) capability(Capability::ImageGatherExtended);
    ImageArguments.push_back(Offset);
  }
  if (Mask) {
    Operands.push_back(Mask); Operands.insert(Operands.end(), ImageArguments.begin(), ImageArguments.end());
  }
  uint32_t Op = Gather ? (Compare ? OpImageDrefGather : OpImageGather) :
      Compare ? (Explicit ? OpImageSampleDrefExplicitLod : OpImageSampleDrefImplicitLod) :
                (Explicit ? OpImageSampleExplicitLod : OpImageSampleImplicitLod);
  uint32_t ResultType = Compare && !Gather ? R.Element : Color;
  return ImageResult(valueOp(Op, ResultType, Operands), ResultType);
}
}
