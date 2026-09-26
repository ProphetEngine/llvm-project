//===-- ProphetSpirvEmitter.cpp - Shared AST module construction ----------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"
#include "clang/Basic/SourceManager.h"
#include "clang/AST/HLSLResource.h"
#include "llvm/ADT/StringExtras.h"
#include <cstring>

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

void SpirvWords::emit(uint32_t Op, ArrayRef<uint32_t> Args) {
  Words.push_back((uint32_t(Args.size() + 1) << 16) | Op);
  Words.insert(Words.end(), Args.begin(), Args.end());
}
void SpirvWords::append(const SpirvWords &Other) {
  Words.insert(Words.end(), Other.Words.begin(), Other.Words.end());
}
void SpirvWords::string(uint32_t Op, ArrayRef<uint32_t> Prefix, StringRef Text,
                        ArrayRef<uint32_t> Suffix) {
  std::vector<uint32_t> Args(Prefix.begin(), Prefix.end());
  size_t Start = Args.size();
  Args.resize(Start + (Text.size() + 4) / 4, 0);
  for (size_t I = 0; I < Text.size(); ++I)
    Args[Start + I / 4] |= uint32_t(uint8_t(Text[I])) << ((I % 4) * 8);
  Args.insert(Args.end(), Suffix.begin(), Suffix.end());
  emit(Op, Args);
}
StringRef shaderAnnotation(const Decl *D, StringRef Prefix) {
  for (const auto *A : D->specific_attrs<AnnotateAttr>())
    if (A->getAnnotation().starts_with(Prefix))
      return A->getAnnotation().drop_front(Prefix.size());
  return {};
}
bool hasShaderAnnotation(const Decl *D, StringRef Name) {
  for (const auto *A : D->specific_attrs<AnnotateAttr>())
    if (A->getAnnotation() == Name)
      return true;
  return false;
}
const ClassTemplateSpecializationDecl *shaderTemplate(QualType T) {
  if (T.isNull()) return nullptr;
  return dyn_cast_or_null<ClassTemplateSpecializationDecl>(
      T.getNonReferenceType()->getAsCXXRecordDecl());
}
QualType shaderTemplateElement(QualType T) {
  if (auto *S = shaderTemplate(T))
    if (S->getTemplateArgs().size() &&
        S->getTemplateArgs()[0].getKind() == TemplateArgument::Type)
      return S->getTemplateArgs()[0].getAsType();
  return {};
}
uint32_t shaderTemplateCount(QualType T) {
  if (auto *S = shaderTemplate(T))
    for (const TemplateArgument &A : S->getTemplateArgs().asArray())
      if (A.getKind() == TemplateArgument::Integral)
        return uint32_t(A.getAsIntegral().getLimitedValue());
  return 0;
}

ProphetSpirvEmitter::ProphetSpirvEmitter(ASTContext &Context, StringRef Entry,
    uint32_t Stage, uint32_t Env, uint32_t Flags, std::string &Error)
    : Ctx(Context), CompileFlags(Flags), EntryName(Entry), RequestedStage(Stage), Environment(Env),
      Diagnostic(Error), CurrentStage(Stage) {}

bool ProphetSpirvEmitter::fail(const llvm::Twine &Message, SourceLocation Loc) {
  if (Diagnostic.empty()) {
    Diagnostic = Message.str();
    if (Loc.isValid()) {
      PresumedLoc P = Ctx.getSourceManager().getPresumedLoc(Loc);
      if (P.isValid())
        Diagnostic = (Twine(P.getFilename()) + ":" + Twine(P.getLine()) + ":" +
                      Twine(P.getColumn()) + ": error: " + Diagnostic).str();
    }
  }
  return false;
}
void ProphetSpirvEmitter::capability(uint32_t Value) {
  if (CapabilitySet.insert(Value).second)
    Capabilities.emit(OpCapability, {Value});
}
void ProphetSpirvEmitter::extension(StringRef Name) {
  if (ExtensionSet.insert(Name.str()).second)
    Extensions.string(OpExtension, {}, Name);
}
void ProphetSpirvEmitter::name(uint32_t Id, StringRef Name) {
  if (!Name.empty()) Debug.string(OpName, {Id}, Name);
}
void ProphetSpirvEmitter::decorate(uint32_t Id, uint32_t Decoration,
                                   ArrayRef<uint32_t> Args) {
  std::vector<uint32_t> Operands{Id, Decoration};
  Operands.insert(Operands.end(), Args.begin(), Args.end());
  Decorations.emit(OpDecorate, Operands);
}
void ProphetSpirvEmitter::memberDecorate(uint32_t Id, uint32_t Member,
    uint32_t Decoration, ArrayRef<uint32_t> Args) {
  std::vector<uint32_t> Operands{Id, Member, Decoration};
  Operands.insert(Operands.end(), Args.begin(), Args.end());
  Decorations.emit(OpMemberDecorate, Operands);
}
uint32_t ProphetSpirvEmitter::valueOp(uint32_t Op, uint32_t T,
                                      ArrayRef<uint32_t> Args) {
  uint32_t Result = id();
  std::vector<uint32_t> Operands{T, Result};
  Operands.insert(Operands.end(), Args.begin(), Args.end());
  Body.emit(Op, Operands);
  trackValue(Result, Op, Args);
  // Preserve source floating operations throughout the shared backend. This
  // conservatively honours HLSL precise, including dependencies across helpers.
  if (Precise) switch (Op) {
  case OpFAdd: case OpFSub: case OpFMul: case OpFDiv: case OpFRem: case OpFMod:
  case OpVectorTimesScalar: case OpMatrixTimesScalar: case OpVectorTimesMatrix:
  case OpMatrixTimesVector: case OpMatrixTimesMatrix: case OpOuterProduct: case OpDot:
    decorate(Result, Decoration::NoContraction); break;
  default: break;
  }
  return Result;
}
uint32_t ProphetSpirvEmitter::extOp(uint32_t Op, uint32_t T,
                                    ArrayRef<uint32_t> Args) {
  if (!Glsl) {
    Glsl = id();
    Imports.string(OpExtInstImport, {Glsl}, "GLSL.std.450");
  }
  std::vector<uint32_t> Operands{Glsl, Op};
  Operands.insert(Operands.end(), Args.begin(), Args.end());
  return valueOp(OpExtInst, T, Operands);
}
void ProphetSpirvEmitter::label(uint32_t Id) {
  Body.emit(OpLabel, {Id});
  CurrentLabel = Id;
  Terminated = false;
}
void ProphetSpirvEmitter::branch(uint32_t Id) {
  if (!Terminated) Body.emit(OpBranch, {Id});
  Terminated = true;
}
void ProphetSpirvEmitter::conditionalBranch(uint32_t Condition, uint32_t Yes,
                                            uint32_t No) {
  Body.emit(OpBranchConditional, {Condition, Yes, No});
  Terminated = true;
}
uint32_t ProphetSpirvEmitter::globalVariable(uint32_t T, uint32_t Storage,
                                             StringRef Name, uint32_t Initializer) {
  uint32_t V = id();
  std::vector<uint32_t> Args{pointerType(T, Storage), V, Storage};
  if (Initializer) Args.push_back(Initializer);
  GlobalVariables.emit(OpVariable, Args);
  AllGlobals.insert(V);
  trackPointer(V, Storage, Initializer);
  name(V, Name);
  return V;
}

void ProphetSpirvEmitter::scan(const DeclContext *DC) {
  for (const Decl *D : DC->decls()) {
    if (!Diagnostic.empty()) return;
    if (const auto *Buffer = dyn_cast<HLSLBufferDecl>(D)) {
      constantBuffer(Buffer); continue;
    }
    if (D->isImplicit() || Ctx.getSourceManager().isInSystemHeader(D->getLocation()))
      continue;
    if (const auto *N = dyn_cast<NamespaceDecl>(D)) scan(N);
    else if (const auto *L = dyn_cast<LinkageSpecDecl>(D)) scan(L);
    else if (const auto *B = dyn_cast<HLSLBufferDecl>(D)) constantBuffer(B);
    else if (const auto *V = dyn_cast<VarDecl>(D)) {
      if (!V->getType()->isDependentType() && !isa<VarTemplateSpecializationDecl>(V))
        global(V);
    } else if (const auto *F = dyn_cast<FunctionDecl>(D)) {
      if (!F->doesThisDeclarationHaveABody() || F->isDependentContext()) continue;
      if (RequestedStage == 9 ? (F->hasAttr<HLSLShaderAttr>() ||
                                 !shaderAnnotation(F, "prophet.stage:").empty())
                              : F->getNameAsString() == EntryName)
        EntryFunctions.push_back(F);
    }
  }
}
// Resolve the task payload before allocating globals, so every access chain
// has its final storage class. Shader parameters and function-local aliases
// may still use the single cached payload at a dispatch call.
void ProphetSpirvEmitter::findTaskPayloads(const DeclContext *DC) {
  std::function<void(const Stmt *)> Visit = [&](const Stmt *S) {
    if (!S) return;
    if (const auto *C = dyn_cast<CallExpr>(S)) {
      const FunctionDecl *F = C->getDirectCallee();
      if (F && F->getNameAsString() == "DispatchMesh" && C->getNumArgs() == 4) {
        const Expr *E = C->getArg(3)->IgnoreParenImpCasts();
        if (const auto *O = dyn_cast<HLSLOutArgExpr>(E)) E = O->getArgLValue()->IgnoreParenImpCasts();
        if (const auto *R = dyn_cast<DeclRefExpr>(E)) {
          if (const auto *V = dyn_cast<VarDecl>(R->getDecl()))
            if (V->hasGlobalStorage() &&
                (hasShaderAnnotation(V, "prophet.groupshared") ||
                 V->getType().getAddressSpace() == LangAS::hlsl_groupshared))
              TaskPayloadDecls.insert(V->getCanonicalDecl());
        }
      }
    }
    for (const Stmt *Child : S->children()) Visit(Child);
  };
  for (const Decl *D : DC->decls()) {
    if (Ctx.getSourceManager().isInSystemHeader(D->getLocation())) continue;
    if (const auto *F = dyn_cast<FunctionDecl>(D)) Visit(F->getBody());
    else if (const auto *N = dyn_cast<NamespaceDecl>(D)) findTaskPayloads(N);
    else if (const auto *L = dyn_cast<LinkageSpecDecl>(D)) findTaskPayloads(L);
  }
  // An entry has exactly one payload interface even with several dispatch sites.
  if (TaskPayloadDecls.size() > 1) TaskPayloadDecls.clear();
}
bool ProphetSpirvEmitter::run(std::vector<uint8_t> &Output) {
  std::function<bool(const Decl *)> PreciseDecl;
  std::function<bool(const Stmt *)> PreciseStmt = [&](const Stmt *S) {
    if (!S) return false;
    if (const auto *DS = dyn_cast<DeclStmt>(S))
      for (const Decl *D : DS->decls()) if (PreciseDecl(D)) return true;
    for (const Stmt *Child : S->children()) if (PreciseStmt(Child)) return true;
    return false;
  };
  PreciseDecl = [&](const Decl *D) {
    if (!D || D->isImplicit() || Ctx.getSourceManager().isInSystemHeader(D->getLocation())) return false;
    if (D->hasAttr<HLSLPreciseAttr>()) return true;
    if (const auto *F = dyn_cast<FunctionDecl>(D)) if (PreciseStmt(F->getBody())) return true;
    if (const auto *DC = dyn_cast<DeclContext>(D))
      for (const Decl *Child : DC->decls()) if (PreciseDecl(Child)) return true;
    return false;
  };
  Precise = PreciseDecl(Ctx.getTranslationUnitDecl());
  capability(Capability::Shader);
  if (RequestedStage == 7) findTaskPayloads(Ctx.getTranslationUnitDecl());
  std::function<void(const DeclContext *)> ReserveBindings = [&](const DeclContext *DC) {
    for (const Decl *D : DC->decls()) {
      if (Ctx.getSourceManager().isInSystemHeader(D->getLocation())) continue;
      if (const auto *N = dyn_cast<NamedDecl>(D)) {
        bool Explicit = clang::hlsl::ResourceBindingAttrs(N).isExplicit();
        for (const auto *A : D->specific_attrs<AnnotateAttr>()) {
          StringRef Name = A->getAnnotation();
          Explicit |= Name.starts_with("prophet.storage_buffer:") || Name.starts_with("prophet.uniform_buffer:") ||
              Name.starts_with("prophet.sampled_texture:") || Name.starts_with("prophet.storage_texture:") ||
              Name.starts_with("prophet.sampler:") || Name.starts_with("prophet.acceleration_structure:");
        }
        if (Explicit) { uint32_t Set, Slot; if (!binding(N, Set, Slot)) return; }
      }
      if (const auto *N = dyn_cast<NamespaceDecl>(D)) ReserveBindings(N);
      else if (const auto *L = dyn_cast<LinkageSpecDecl>(D)) ReserveBindings(L);
    }
  };
  ReserveBindings(Ctx.getTranslationUnitDecl());
  if (!Diagnostic.empty()) return false;
  scan(Ctx.getTranslationUnitDecl());
  if (!Diagnostic.empty()) return false;
  if (EntryFunctions.empty()) return fail("shader entry point was not found");
  if (RequestedStage != 9 && EntryFunctions.size() != 1)
    return fail("shader entry point is ambiguous");
  for (const FunctionDecl *F : EntryFunctions)
    if (!entry(F)) return false;
  return validateUniformity() && finish(Output);
}
bool ProphetSpirvEmitter::finish(std::vector<uint8_t> &Output) {
  if (!Diagnostic.empty()) return false;
  SpirvWords Module;
  Module.Words = {0x07230203u, Environment == 1 ? 0x00010300u : 0x00010500u,
                  0x50520002u, NextId, 0};
  Module.append(Capabilities);
  Module.append(Extensions);
  Module.append(Imports);
  Module.emit(OpMemoryModel, {AddressingModel::Logical, MemoryModel::GLSL450});
  Module.append(Entries);
  Module.append(Modes);
  Module.append(Debug);
  Module.append(Decorations);
  Module.append(Declarations);
  Module.append(GlobalVariables);
  Module.append(FunctionDeclarations);
  Module.append(Functions);
  Output.resize(Module.Words.size() * 4);
  for (size_t I = 0; I < Module.Words.size(); ++I)
    for (unsigned B = 0; B < 4; ++B)
      Output[I * 4 + B] = uint8_t(Module.Words[I] >> (8 * B));
  return true;
}
bool emitShaderSpirv(ASTContext &Context, StringRef EntryPoint, uint32_t Stage,
                     uint32_t Environment, uint32_t Flags, std::vector<uint8_t> &Output,
                     std::string &Diagnostic) {
  Output.clear();
  ProphetSpirvEmitter Emitter(Context, EntryPoint, Stage, Environment, Flags, Diagnostic);
  return Emitter.run(Output);
}
}
