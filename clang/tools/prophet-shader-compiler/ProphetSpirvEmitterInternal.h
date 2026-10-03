//===-- ProphetSpirvEmitterInternal.h - Typed AST lowering state ----------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PROPHET_SPIRV_EMITTER_INTERNAL_H
#define PROPHET_SPIRV_EMITTER_INTERNAL_H

#include "ProphetSpirvEmitter.h"
#include "ProphetSpirvEnums.h"
#include "ProphetShaderContracts.h"
#include "clang/AST/Attr.h"
#include "clang/AST/DeclCXX.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/AST/ExprCXX.h"
#include "clang/AST/StmtCXX.h"
#include "llvm/ADT/ArrayRef.h"
#include <map>
#include <functional>
#include <memory>
#include <set>
#include <tuple>

namespace prophet {
using clang::QualType;
using llvm::StringRef;

struct SpirvWords {
  std::vector<uint32_t> Words;
  void emit(uint32_t Op, llvm::ArrayRef<uint32_t> Args = {});
  void append(const SpirvWords &Other);
  void string(uint32_t Op, llvm::ArrayRef<uint32_t> Prefix, StringRef Text,
              llvm::ArrayRef<uint32_t> Suffix = {});
};

enum class SpirvLayout { Natural, Storage, Uniform };
struct SpirvType {
  enum Kind { Void, Bool, Int, Float, Vector, Matrix, Array, RuntimeArray,
              Struct, Image, Sampler, Acceleration } K = Void;
  uint32_t Bits = 0, Element = 0, Count = 0, Size = 0, Align = 1, MatrixStride = 0;
  bool Signed = false, RowMajor = false, NativeMatrix = false;
  uint32_t Rows = 0;
  std::vector<uint32_t> Members;
  std::vector<uint32_t> Offsets;
};

struct SpirvResource;
struct SpirvClosure;
struct SpirvValue {
  SpirvValue() = default;
  SpirvValue(uint32_t I, uint32_t T, uint32_t S = 7, QualType A = {}, bool P = false)
      : Id(I), Type(T), Storage(S), ASTType(A), Address(P) {}
  uint32_t Id = 0, Type = 0, Storage = 7;
  QualType ASTType;
  bool Address = false;
  SpirvResource *Resource = nullptr;
  std::shared_ptr<SpirvClosure> Closure;
  uint32_t ImageCoordinate = 0;
  bool DescriptorSelected = false, Stream = false;
  const clang::ValueDecl *Interface = nullptr;
  std::vector<uint32_t> InterfacePath;
  std::vector<uint32_t> Swizzle;
  std::vector<std::pair<uint32_t, uint32_t>> MatrixComponents;
  std::vector<uint32_t> Path;
  explicit operator bool() const { return Id || Resource || Closure || Stream || Interface; }
};
struct SpirvClosure {
  const clang::CXXRecordDecl *Record = nullptr;
  std::map<const clang::ValueDecl *, SpirvValue> Captures;
  std::set<const clang::ValueDecl *> Borrowed;
};
struct SpirvResource {
  enum Kind { Buffer, Texture, StorageTexture, Sampler, ComparisonSampler,
              AccelerationStructure } K = Buffer;
  const clang::VarDecl *Decl = nullptr;
  QualType ElementType;
  uint32_t Variable = 0, Type = 0, Element = 0, Set = 0, Binding = 0;
  uint32_t Dimension = 1, Count = 1;
  bool Arrayed = false, Multisample = false, Depth = false, ReadOnly = false;
  bool CppRecord = false, Uniform = false;
};
struct SpirvInterface {
  const clang::ValueDecl *Decl = nullptr;
  QualType ASTType;
  std::vector<uint32_t> Path;
  uint32_t Variable = 0, Type = 0, Storage = 1, ArraySize = 0;
  bool Patch = false, Primitive = false;
  std::string Semantic;
};

class ProphetSpirvEmitter {
  clang::ASTContext &Ctx;
  uint32_t CompileFlags;
  StringRef EntryName;
  uint32_t RequestedStage, Environment;
  std::string &Diagnostic;
  SpirvWords Capabilities, Extensions, Imports, Entries, Modes, Debug,
      Decorations, Declarations, GlobalVariables, Functions;
  SpirvWords LocalVariables, Body;
  uint32_t NextId = 1, CurrentStage = 1, CurrentEntry = 0, CurrentLabel = 0;
  uint32_t Glsl = 0;
  bool Terminated = false, Precise = false;
  std::set<uint32_t> CapabilitySet;
  std::set<std::string> ExtensionSet;
  std::map<std::vector<uint32_t>, uint32_t> TypeKeys, ConstantKeys;
  std::map<uint32_t, SpirvType> Types;
  std::map<std::tuple<const void *, SpirvLayout, bool>, uint32_t> ASTTypes;
  std::map<const clang::ValueDecl *, SpirvValue> Globals, Locals;
  std::vector<std::unique_ptr<SpirvResource>> Resources;
  std::map<std::pair<uint32_t, uint32_t>, const clang::Decl *> Bindings;
  std::vector<const clang::FunctionDecl *> EntryFunctions;
  std::vector<const clang::VarDecl *> GlobalSemanticDecls;
  std::vector<SpirvInterface> Interfaces;
  std::set<uint32_t> UsedGlobals, AllGlobals;
  std::map<std::tuple<uint32_t, uint32_t, uint32_t>, uint32_t> Builtins;
  std::vector<std::pair<uint32_t, uint32_t>> FlowTargets;
  std::map<uint32_t, size_t> FlowCleanupDepth;
  struct Cleanup { SpirvValue Object; QualType Type; };
  std::vector<Cleanup> Cleanups;
  bool needsDestruction(QualType T);
  bool destroy(SpirvValue Object, QualType T);
  bool destroyMembers(SpirvValue Object, const clang::CXXRecordDecl *R);
  bool emitCleanups(size_t Depth);
  bool scopedStatement(const clang::Stmt *S);
  using UniformLoop = std::shared_ptr<std::vector<uint32_t>>;
  struct UniformSite {
    uint64_t Destination = 0;
    clang::SourceLocation Location;
    std::vector<uint32_t> Conditions;
    std::vector<UniformLoop> Loops;
  };
  uint32_t WorkgroupInvocations = 1;
  std::vector<uint32_t> ControlConditions;
  std::vector<UniformLoop> UniformLoops;
  std::vector<UniformSite> UniformStores, UniformBarriers;
  std::map<uint32_t, uint32_t> PointerRoots, PointerStorage;
  std::map<uint64_t, std::vector<uint64_t>> UniformDependencies;
  std::set<uint64_t> VaryingSources;
  uint32_t pointerRoot(uint32_t Pointer) const;
  void trackPointer(uint32_t Pointer, uint32_t Storage, uint32_t Initializer = 0);
  void trackBuiltin(uint32_t Pointer, uint32_t Builtin);
  void trackValue(uint32_t Result, uint32_t Op, llvm::ArrayRef<uint32_t> Args);
  UniformSite uniformSite() const;
  void trackStore(uint32_t Pointer, uint32_t Value);
  void trackBarrier(clang::SourceLocation Location);
  void trackReturn();
  void trackLoopExit();
  bool validateUniformity();
  struct CallFrame {
    const clang::FunctionDecl *Function = nullptr;
    SpirvValue Result;
    uint32_t Merge = 0;
    size_t CleanupDepth = 0;
    std::vector<uint32_t> ReturnConditions;
    std::vector<UniformLoop> ReturnLoops;
    SpirvValue This;
    std::shared_ptr<SpirvClosure> Closure;
  };
  std::vector<CallFrame> Calls;
  uint32_t NextInput = 0, NextOutput = 0;
  uint32_t GeometryStream = 0, MeshVertices = 0, MeshIndices = 0;
  SpirvValue MeshPayload;
  std::map<std::pair<uint32_t, uint32_t>, SpirvValue> RaySlots;
  SpirvValue raySlot(SpirvValue Value, uint32_t Storage, StringRef Name);
  std::set<const clang::VarDecl *> TaskPayloadDecls;
  void findTaskPayloads(const clang::DeclContext *DC);
  const clang::FunctionDecl *CurrentFunction = nullptr;
  bool ObjectMode = false;
  std::string ObjectModuleId;
  struct ObjectFunction {
    const clang::FunctionDecl *Decl = nullptr;
    const clang::VarDecl *TableDecl = nullptr;
    uint32_t Id = 0, ReturnType = 0, FunctionType = 0;
    std::vector<uint32_t> Parameters;
    std::string StableId, Role, Name, Target, DefaultTarget;
  };
  std::map<const clang::FunctionDecl *, size_t> ObjectFunctionIndices;
  std::map<std::pair<const clang::VarDecl *, const clang::FunctionDecl *>, size_t> ObjectTableFunctionIndices;
  std::vector<ObjectFunction> ObjectFunctions;
  std::map<uint32_t, const clang::NamedDecl *> ObjectResourceOwners;
  SpirvWords FunctionDeclarations;
  void scanObjectFunctions(const clang::DeclContext *DC);
  bool registerObjectFunction(const clang::FunctionDecl *F);
  bool emitObjectFunction(size_t Index);
  bool isObjectBoundary(const clang::FunctionDecl *F) const;
  SpirvValue objectCall(const clang::FunctionDecl *F, llvm::ArrayRef<SpirvValue> Args);
  SpirvValue objectTableCall(const clang::VarDecl *Table, const clang::FunctionDecl *Method,
                             llvm::ArrayRef<SpirvValue> Args);
  uint32_t valueAbiType(QualType T);
  SpirvValue valueAbiConvert(SpirvValue V, QualType T, bool ToAbi);

  uint32_t id() { return NextId++; }
  bool fail(const llvm::Twine &Message, clang::SourceLocation Loc = {});
  void capability(uint32_t Value);
  void extension(StringRef Name);
  void name(uint32_t Id, StringRef Name);
  void decorate(uint32_t Id, uint32_t Decoration, llvm::ArrayRef<uint32_t> Args = {});
  void memberDecorate(uint32_t Id, uint32_t Member, uint32_t Decoration,
                      llvm::ArrayRef<uint32_t> Args = {});
  uint32_t internType(uint32_t Op, llvm::ArrayRef<uint32_t> Args, SpirvType Info);
  uint32_t type(QualType T, SpirvLayout Layout = SpirvLayout::Natural);
  uint32_t intrinsicType(QualType T);
  SpirvType physicalType(QualType T, SpirvLayout Layout);
  SpirvType physicalType(uint32_t T, SpirvLayout Layout, bool RowMajor = false);
  uint32_t integerType(uint32_t Bits = 32, bool Signed = false);
  uint32_t floatType(uint32_t Bits = 32);
  uint32_t boolType();
  uint32_t voidType();
  uint32_t vectorType(uint32_t Element, uint32_t Count);
  uint32_t arrayType(uint32_t Element, uint32_t Count, SpirvLayout Layout, bool Runtime = false, const SpirvType *ElementLayout = nullptr);
  uint32_t structType(llvm::ArrayRef<uint32_t> Members, SpirvLayout Layout, bool Block = false, llvm::ArrayRef<SpirvType> MemberLayouts = {}, llvm::ArrayRef<uint32_t> ExplicitOffsets = {});
  uint32_t pointerType(uint32_t T, uint32_t Storage);
  uint32_t constant(uint32_t T, uint64_t Bits);
  uint32_t constantBool(bool V);
  uint32_t zero(uint32_t T);
  uint32_t composite(uint32_t T, llvm::ArrayRef<uint32_t> Values, bool Constant = false);
  uint32_t constantValue(QualType T, const clang::APValue &V);
  SpirvValue constantExpr(const clang::Expr *E);
  uint32_t valueOp(uint32_t Op, uint32_t T, llvm::ArrayRef<uint32_t> Args);
  uint32_t extOp(uint32_t Op, uint32_t T, llvm::ArrayRef<uint32_t> Args);
  SpirvValue address(SpirvValue V);
  SpirvValue load(SpirvValue V);
  bool store(SpirvValue Target, SpirvValue Value);
  SpirvValue convert(SpirvValue V, uint32_t T, QualType AST = {});
  uint32_t repack(uint32_t Value, uint32_t From, uint32_t To);
  SpirvValue temporary(QualType T, StringRef Name = {});
  SpirvValue temporaryType(uint32_t T, QualType AST = {}, StringRef Name = {});
  SpirvValue access(SpirvValue Base, llvm::ArrayRef<uint32_t> Indices,
                    QualType Result, uint32_t ResultType = 0);
  SpirvValue extract(SpirvValue Base, llvm::ArrayRef<uint32_t> Indices, QualType T);
  uint32_t matrixScalarType(uint32_t T);
  SpirvValue matrixAddress(SpirvValue Base, uint32_t Row, uint32_t Column, QualType Scalar);
  SpirvValue matrixElement(SpirvValue Base, uint32_t Row, uint32_t Column);
  SpirvValue matrixTranspose(SpirvValue Value, QualType Result);
  SpirvValue matrixMultiply(SpirvValue A, SpirvValue B, QualType Result);
  std::vector<uint32_t> ArrayInitIndices;
  std::map<const clang::OpaqueValueExpr *, SpirvValue> OpaqueValues;
  SpirvValue expression(const clang::Expr *E);
  SpirvValue binary(const clang::BinaryOperator *E);
  SpirvValue unary(const clang::UnaryOperator *E);
  SpirvValue arithmetic(clang::BinaryOperatorKind Op, SpirvValue A,
                        SpirvValue B, QualType Result, clang::SourceLocation Loc);
  SpirvValue initialize(QualType T, llvm::ArrayRef<const clang::Expr *> Values);
  SpirvValue copyClosure(SpirvValue V);
  SpirvValue call(const clang::CallExpr *E);
  SpirvValue callFunction(const clang::FunctionDecl *F,
                          llvm::ArrayRef<SpirvValue> Args, SpirvValue Object = {});
  SpirvValue intrinsic(StringRef Name, const clang::CallExpr *E,
                       llvm::ArrayRef<SpirvValue> Args, bool &Handled);
  SpirvValue resourceCall(StringRef Name, SpirvValue Object, const clang::CallExpr *E,
                          llvm::ArrayRef<SpirvValue> Args, bool &Handled);
  SpirvValue imageCall(StringRef Name, SpirvResource &R, const clang::CallExpr *E,
                       llvm::ArrayRef<SpirvValue> Args);
  SpirvValue resourceIndex(SpirvValue Object, SpirvValue Index, QualType T);
  bool statement(const clang::Stmt *S);
  bool declaration(const clang::Decl *D);
  bool conditional(const clang::IfStmt *S);
  bool loop(const clang::Stmt *S);
  bool rangeFor(const clang::CXXForRangeStmt *S);
  bool switchStmt(const clang::SwitchStmt *S);
  void label(uint32_t Id);
  void branch(uint32_t Id);
  void conditionalBranch(uint32_t Condition, uint32_t Yes, uint32_t No);
  void scan(const clang::DeclContext *DC);
  bool global(const clang::VarDecl *D);
  bool resource(const clang::VarDecl *D);
  bool constantBuffer(const clang::HLSLBufferDecl *D);
  bool binding(const clang::NamedDecl *D, uint32_t &Set, uint32_t &Binding);
  uint32_t globalVariable(uint32_t T, uint32_t Storage, StringRef Name, uint32_t Initializer = 0);
  uint32_t builtin(uint32_t Builtin, uint32_t T, uint32_t Storage = 1);
  uint32_t stage(const clang::FunctionDecl *F) const;
  std::string semantic(const clang::Decl *D) const;
  uint32_t semanticIndex(StringRef Semantic) const;
  bool interface(const clang::ValueDecl *D, QualType T, uint32_t Storage,
                  std::vector<uint32_t> Path = {}, uint32_t ArraySize = 0,
                  bool Patch = false, bool Primitive = false, StringRef OverrideSemantic = {});
  bool entry(const clang::FunctionDecl *F);
  bool regularEntry(const clang::FunctionDecl *F);
  bool geometryEntry(const clang::FunctionDecl *F);
  uint32_t isolatedFunction(StringRef Name, const std::function<bool()> &Emit);
  bool hullEntry(const clang::FunctionDecl *F);
  bool domainEntry(const clang::FunctionDecl *F);
  bool meshEntry(const clang::FunctionDecl *F);
  bool rayEntry(const clang::FunctionDecl *F);
  QualType componentType(QualType T, uint32_t Index);
  SpirvValue interfaceReference(const clang::ValueDecl *D, QualType T, uint32_t Storage);
  SpirvValue interfaceAddress(SpirvValue V);
  SpirvValue loadInterface(SpirvValue V);
  bool storeInterface(SpirvValue Target, SpirvValue Value);
  SpirvValue interfaceInput(const clang::ValueDecl *D, QualType T, uint32_t Index = 0);
  bool interfaceOutput(const clang::ValueDecl *D, SpirvValue V, uint32_t Index = 0);
  uint32_t executionModel(uint32_t Stage);
  void executionMode(uint32_t Mode, llvm::ArrayRef<uint32_t> Args = {});
  bool finish(std::vector<uint8_t> &Output);

public:
  ProphetSpirvEmitter(clang::ASTContext &Context, StringRef Entry, uint32_t Stage,
                      uint32_t Environment, uint32_t Flags, std::string &Diagnostic);
  bool run(std::vector<uint8_t> &Output);
  bool runObject(llvm::StringRef ModuleId, std::vector<uint8_t> &Output,
                 std::string &Manifest);
};

StringRef shaderAnnotation(const clang::Decl *D, StringRef Prefix);
bool hasShaderAnnotation(const clang::Decl *D, StringRef Name);
const clang::ClassTemplateSpecializationDecl *shaderTemplate(QualType T);
QualType shaderTemplateElement(QualType T);
uint32_t shaderTemplateCount(QualType T);
}
#endif
