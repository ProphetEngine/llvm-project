//===-- ProphetSpirvEmitter.Statements.cpp - Structured AST control -------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetSpirvEmitterInternal.h"

using namespace clang;
using namespace llvm;
namespace prophet {
using namespace spv;

bool ProphetSpirvEmitter::needsDestruction(QualType T) {
  if (T->isReferenceType() || T->isPointerType()) return false;
  while (const auto *A = Ctx.getAsArrayType(T)) T = A->getElementType();
  const auto *R = T->getAsCXXRecordDecl();
  return R && !R->hasTrivialDestructor();
}
bool ProphetSpirvEmitter::destroyMembers(SpirvValue Object, const CXXRecordDecl *R) {
  std::vector<const FieldDecl *> Fields(R->field_begin(), R->field_end());
  for (auto I = Fields.rbegin(); I != Fields.rend(); ++I) {
    const FieldDecl *F = *I;
    if (needsDestruction(F->getType()))
      if (!destroy(access(Object, {constant(integerType(), R->getNumBases() + F->getFieldIndex())}, F->getType()), F->getType())) return false;
  }
  std::vector<QualType> Bases;
  for (const auto &B : R->bases()) Bases.push_back(B.getType());
  for (size_t I = Bases.size(); I > 0; --I)
    if (needsDestruction(Bases[I - 1]))
      if (!destroy(access(Object, {constant(integerType(), I - 1)}, Bases[I - 1]), Bases[I - 1])) return false;
  return Diagnostic.empty();
}
bool ProphetSpirvEmitter::destroy(SpirvValue Object, QualType T) {
  if (!needsDestruction(T)) return true;
  if (const auto *A = Ctx.getAsConstantArrayType(T)) {
    for (uint64_t I = A->getSize().getLimitedValue(); I > 0; --I)
      if (!destroy(access(Object, {constant(integerType(), I - 1)}, A->getElementType()), A->getElementType())) return false;
    return true;
  }
  const auto *R = T->getAsCXXRecordDecl();
  const auto *Dtor = R->getDestructor();
  if (Dtor && Dtor->hasBody()) {
    callFunction(Dtor, {}, Object);
    return Diagnostic.empty();
  }
  return destroyMembers(Object, R);
}
bool ProphetSpirvEmitter::emitCleanups(size_t Depth) {
  // Copy each descriptor: destructor calls may append their own local cleanups.
  for (size_t I = Cleanups.size(); I > Depth; --I) {
    Cleanup C = Cleanups[I - 1];
    if (!destroy(C.Object, C.Type)) return false;
  }
  return Diagnostic.empty();
}
bool ProphetSpirvEmitter::scopedStatement(const Stmt *S) {
  size_t Depth = Cleanups.size();
  bool Result = statement(S);
  if (Result && !Terminated) Result = emitCleanups(Depth);
  Cleanups.resize(Depth);
  return Result;
}
bool ProphetSpirvEmitter::declaration(const Decl *D) {
  const auto *V = dyn_cast<VarDecl>(D);
  if (!V) return isa<TypeDecl>(D) || isa<StaticAssertDecl>(D) || isa<UsingDecl>(D);
  if (V->isStaticLocal()) {
    if (needsDestruction(V->getType())) return fail("static shader objects must have trivial destruction", V->getLocation());
    return global(V);
  }
  if (V->getType()->isReferenceType()) {
    if (!V->hasInit()) return fail("shader reference requires an initializer", V->getLocation());
    SpirvValue Init = expression(V->getInit());
    if (!Init.Closure && !Init.Resource && !Init.Address && !Init.Stream && !Init.Interface) {
      if (V->getInit()->IgnoreParenImpCasts()->isGLValue())
        return fail("runtime-selected reference requires a statically known shader address", V->getLocation());
      SpirvValue Copy = temporary(V->getType().getNonReferenceType());
      if (!store(Copy, Init)) return false;
      Init = Copy;
    }
    Locals[V->getCanonicalDecl()] = Init;
    return bool(Init);
  }
  if (const auto *R = V->getType()->getAsCXXRecordDecl(); R && R->isLambda()) {
    Locals[V->getCanonicalDecl()] = expression(V->getInit());
    return bool(Locals[V->getCanonicalDecl()]);
  }
  if (V->getType()->isPointerType()) {
    if (!V->hasInit()) return fail("shader pointer requires a known object", V->getLocation());
    Locals[V->getCanonicalDecl()] = expression(V->getInit());
    return bool(Locals[V->getCanonicalDecl()]);
  }
  SpirvValue Address = temporary(V->getType(), V->getName());
  Locals[V->getCanonicalDecl()] = Address;
  if (V->hasInit() && !store(Address, expression(V->getInit()))) return false;
  if (needsDestruction(V->getType())) Cleanups.push_back({Address, V->getType()});
  return Diagnostic.empty();
}
bool ProphetSpirvEmitter::statement(const Stmt *S) {
  if (!S || Terminated) return true;
  if (!Diagnostic.empty()) return false;
  if (const auto *C = dyn_cast<CompoundStmt>(S)) {
    size_t Depth = Cleanups.size();
    for (const Stmt *Child : C->body())
      if (!statement(Child)) return false;
    if (!Terminated && !emitCleanups(Depth)) return false;
    Cleanups.resize(Depth);
    return true;
  }
  if (const auto *D = dyn_cast<DeclStmt>(S)) {
    for (const Decl *Child : D->decls()) if (!declaration(Child)) return false;
    return true;
  }
  if (const auto *I = dyn_cast<IfStmt>(S)) {
    size_t Depth = Cleanups.size();
    bool Result = conditional(I);
    if (Result && !Terminated) Result = emitCleanups(Depth);
    Cleanups.resize(Depth); return Result;
  }
  if (isa<ForStmt>(S) || isa<WhileStmt>(S) || isa<DoStmt>(S)) return loop(S);
  if (isa<CXXForRangeStmt>(S) || isa<SwitchStmt>(S)) {
    size_t Depth = Cleanups.size();
    bool Result = isa<CXXForRangeStmt>(S) ? rangeFor(cast<CXXForRangeStmt>(S)) : switchStmt(cast<SwitchStmt>(S));
    if (Result && !Terminated) Result = emitCleanups(Depth);
    Cleanups.resize(Depth); return Result;
  }
  if (const auto *R = dyn_cast<ReturnStmt>(S)) {
    if (Calls.empty()) return fail("return outside a shader function", S->getBeginLoc());
    if (const Expr *E = R->getRetValue()) {
      SpirvValue V = expression(E);
      if (Calls.back().Function->getReturnType()->isReferenceType())
        return fail("runtime helper reference returns require a statically known alias", E->getExprLoc());
      if (Calls.back().Result && !store(Calls.back().Result, V)) return false;
    }
    if (!emitCleanups(Calls.back().CleanupDepth)) return false;
    trackReturn(); branch(Calls.back().Merge); return true;
  }
  if (isa<BreakStmt>(S)) {
    if (FlowTargets.empty()) return fail("break outside a structured construct", S->getBeginLoc());
    if (FlowTargets.back().second) trackLoopExit();
    if (!emitCleanups(FlowCleanupDepth[FlowTargets.back().first])) return false;
    branch(FlowTargets.back().first); return true;
  }
  if (isa<ContinueStmt>(S)) {
    for (auto I = FlowTargets.rbegin(); I != FlowTargets.rend(); ++I)
      if (I->second) {
        uint32_t Target = I->second;
        if (!emitCleanups(FlowCleanupDepth[Target])) return false;
        trackLoopExit(); branch(Target); return true;
      }
    return fail("continue outside a loop", S->getBeginLoc());
  }
  if (const auto *A = dyn_cast<AttributedStmt>(S)) return statement(A->getSubStmt());
  if (const auto *L = dyn_cast<LabelStmt>(S)) return statement(L->getSubStmt());
  if (isa<NullStmt>(S)) return true;
  if (const auto *E = dyn_cast<Expr>(S)) { expression(E); return Diagnostic.empty(); }
  return fail(("unsupported shader statement: " + Twine(S->getStmtClassName())).str(), S->getBeginLoc());
}
bool ProphetSpirvEmitter::conditional(const IfStmt *S) {
  if (!statement(S->getInit())) return false;
  if (S->getConditionVariable() && !declaration(S->getConditionVariable())) return false;
  if (S->isConstexpr()) {
    Expr::EvalResult R;
    if (S->getCond()->EvaluateAsInt(R, Ctx))
      return scopedStatement(R.Val.getInt().getBoolValue() ? S->getThen() : S->getElse());
  }
  SpirvValue Condition = convert(expression(S->getCond()), boolType());
  if (!Condition) return false;
  uint32_t Yes = id(), No = id(), Merge = id();
  Body.emit(OpSelectionMerge, {Merge, 0});
  conditionalBranch(Condition.Id, Yes, No);
  ControlConditions.push_back(Condition.Id);
  label(Yes); if (!scopedStatement(S->getThen())) return false; branch(Merge);
  label(No); if (!scopedStatement(S->getElse())) return false; branch(Merge);
  ControlConditions.pop_back();
  label(Merge); return true;
}
bool ProphetSpirvEmitter::loop(const Stmt *S) {
  size_t Depth = Cleanups.size();
  const Expr *Condition = nullptr, *Increment = nullptr;
  const Stmt *LoopBody = nullptr;
  bool IsDo = isa<DoStmt>(S);
  if (const auto *F = dyn_cast<ForStmt>(S)) {
    if (!statement(F->getInit())) return false;
    if (F->getConditionVariable() && needsDestruction(F->getConditionVariable()->getType()))
      return fail("loop condition objects must have trivial destruction", F->getForLoc());
    Condition = F->getCond(); Increment = F->getInc(); LoopBody = F->getBody();
  } else if (const auto *W = dyn_cast<WhileStmt>(S)) {
    if (W->getConditionVariable() && needsDestruction(W->getConditionVariable()->getType()))
      return fail("loop condition objects must have trivial destruction", W->getWhileLoc());
    Condition = W->getCond(); LoopBody = W->getBody();
  } else if (const auto *D = dyn_cast<DoStmt>(S)) {
    Condition = D->getCond(); LoopBody = D->getBody();
  }
  uint32_t Header = id(), Check = id(), Block = id(), Continue = id(), Merge = id();
  auto UniformGuard = std::make_shared<std::vector<uint32_t>>(); UniformLoops.push_back(UniformGuard);
  branch(Header); label(Header);
  Body.emit(OpLoopMerge, {Merge, Continue, 0}); branch(IsDo ? Block : Check);
  if (!IsDo) {
    label(Check);
    const VarDecl *Variable = nullptr;
    if (const auto *W = dyn_cast<WhileStmt>(S)) Variable = W->getConditionVariable();
    if (const auto *F = dyn_cast<ForStmt>(S)) Variable = F->getConditionVariable();
    if (Variable && !declaration(Variable)) return false;
    uint32_t C = Condition ? convert(expression(Condition), boolType()).Id : constantBool(true);
    UniformGuard->push_back(C); conditionalBranch(C, Block, Merge);
  }
  label(Block); FlowTargets.emplace_back(Merge, Continue);
  FlowCleanupDepth[Merge] = FlowCleanupDepth[Continue] = Cleanups.size();
  if (!scopedStatement(LoopBody)) return false;
  FlowTargets.pop_back(); branch(Continue); label(Continue);
  if (Increment) expression(Increment);
  if (IsDo) {
    uint32_t C = convert(expression(Condition), boolType()).Id; UniformGuard->push_back(C);
    conditionalBranch(C, Header, Merge);
  }
  else branch(Header);
  UniformLoops.pop_back();
  label(Merge);
  bool Result = emitCleanups(Depth); Cleanups.resize(Depth);
  return Result;
}
bool ProphetSpirvEmitter::rangeFor(const CXXForRangeStmt *S) {
  if (!statement(S->getInit())) return false;
  SpirvValue Range = expression(S->getRangeInit());
  const auto *A = Ctx.getAsConstantArrayType(S->getRangeInit()->getType());
  if (!A || !Range.Address) return fail("shader range-for requires a fixed-size array", S->getForLoc());
  const VarDecl *Variable = S->getLoopVariable();
  if (needsDestruction(Variable->getType()))
    return fail("range-for elements with nontrivial destruction must be borrowed by reference", Variable->getLocation());
  SpirvValue Index = temporary(Ctx.UnsignedIntTy);
  store(Index, {constant(integerType(), 0), integerType()});
  uint32_t Header = id(), Check = id(), Block = id(), Continue = id(), Merge = id();
  branch(Header); label(Header); Body.emit(OpLoopMerge, {Merge, Continue, 0}); branch(Check);
  label(Check);
  uint32_t I = load(Index).Id;
  uint32_t Condition = valueOp(OpULessThan, boolType(),
      {I, constant(integerType(), A->getSize().getLimitedValue())});
  auto UniformGuard = std::make_shared<std::vector<uint32_t>>(std::initializer_list<uint32_t>{Condition});
  UniformLoops.push_back(UniformGuard);
  conditionalBranch(Condition, Block, Merge); label(Block);
  SpirvValue Element = access(Range, {I}, A->getElementType());
  if (Variable->getType()->isReferenceType()) Locals[Variable->getCanonicalDecl()] = Element;
  else {
    SpirvValue Copy = temporary(Variable->getType());
    store(Copy, Element); Locals[Variable->getCanonicalDecl()] = Copy;
  }
  FlowTargets.emplace_back(Merge, Continue);
  FlowCleanupDepth[Merge] = FlowCleanupDepth[Continue] = Cleanups.size();
  if (!scopedStatement(S->getBody())) return false;
  FlowTargets.pop_back(); branch(Continue); label(Continue);
  store(Index, {valueOp(OpIAdd, integerType(), {load(Index).Id, constant(integerType(), 1)}), integerType()});
  branch(Header); UniformLoops.pop_back(); label(Merge); return Diagnostic.empty();
}
bool ProphetSpirvEmitter::switchStmt(const SwitchStmt *S) {
  if (!statement(S->getInit())) return false;
  if (S->getConditionVariable() && !declaration(S->getConditionVariable())) return false;
  SpirvValue Selector = load(expression(S->getCond()));
  struct Block { const SwitchCase *Case; uint32_t Label; std::vector<const Stmt *> Statements; };
  std::vector<Block> Blocks;
  std::function<void(const Stmt *)> Collect = [&](const Stmt *Node) {
    if (const auto *C = dyn_cast<SwitchCase>(Node)) {
      Blocks.push_back({C, id(), {}}); Collect(C->getSubStmt());
    } else if (const auto *Compound = dyn_cast<CompoundStmt>(Node); Compound && Blocks.empty()) {
      for (const Stmt *Child : Compound->body()) Collect(Child);
    } else if (!Blocks.empty()) Blocks.back().Statements.push_back(Node);
  };
  // Top-level statements belong to the most recent case, including fallthrough.
  if (const auto *C = dyn_cast<CompoundStmt>(S->getBody()))
    for (const Stmt *Child : C->body()) Collect(Child);
  else Collect(S->getBody());
  uint32_t Merge = id(), Default = Merge;
  for (const auto &B : Blocks) if (isa<DefaultStmt>(B.Case)) Default = B.Label;
  std::vector<uint32_t> Args{Selector.Id, Default};
  for (const auto &B : Blocks) {
    if (const auto *C = dyn_cast<CaseStmt>(B.Case)) {
      if (C->getRHS()) return fail("case ranges require explicit shader cases", C->getBeginLoc());
      Expr::EvalResult R;
      if (!C->getLHS()->EvaluateAsInt(R, Ctx)) return fail("case label is not constant", C->getBeginLoc());
      uint64_t V = R.Val.getInt().getLimitedValue(); Args.push_back(uint32_t(V));
      if (Types[Selector.Type].Bits == 64) Args.push_back(uint32_t(V >> 32));
      Args.push_back(B.Label);
    }
  }
  Body.emit(OpSelectionMerge, {Merge, 0}); Body.emit(OpSwitch, Args); Terminated = true;
  ControlConditions.push_back(Selector.Id);
  FlowTargets.emplace_back(Merge, 0);
  FlowCleanupDepth[Merge] = Cleanups.size();
  for (size_t I = 0; I < Blocks.size(); ++I) {
    label(Blocks[I].Label);
    for (const Stmt *Node : Blocks[I].Statements) {
      if (const auto *DS = dyn_cast<DeclStmt>(Node))
        for (const Decl *D : DS->decls())
          if (const auto *V = dyn_cast<VarDecl>(D); V && needsDestruction(V->getType()))
            return fail("put a shader case object with a destructor inside a braced case scope", V->getLocation());
      if (!statement(Node)) return false;
    }
    branch(I + 1 < Blocks.size() ? Blocks[I + 1].Label : Merge);
  }
  FlowTargets.pop_back(); ControlConditions.pop_back(); label(Merge); return true;
}
SpirvValue ProphetSpirvEmitter::callFunction(const FunctionDecl *F,
    ArrayRef<SpirvValue> Args, SpirvValue Object) {
  if (ObjectMode && isObjectBoundary(F) &&
      (!Calls.empty() || !CurrentFunction || F->getCanonicalDecl() != CurrentFunction->getCanonicalDecl()))
    return objectCall(F, Args);
  const FunctionDecl *Definition = F->getDefinition();
  if (!Definition) { fail("shader function has no body: " + F->getNameAsString(), F->getLocation()); return {}; }
  F = Definition;
  for (const auto &Frame : Calls)
    if (Frame.Function->getCanonicalDecl() == F->getCanonicalDecl()) {
      fail("runtime recursive call has no legal Vulkan Shader SPIR-V representation", F->getLocation()); return {};
    }
  if (Args.size() != F->getNumParams()) { fail("shader helper argument count mismatch", F->getLocation()); return {}; }
  auto SavedLocals = Locals;
  auto SavedFlow = FlowTargets;
  FlowTargets.clear();
  CallFrame Frame; Frame.Function = F; Frame.Merge = id(); Frame.This = Object;
  Frame.CleanupDepth = Cleanups.size();
  if (needsDestruction(F->getReturnType())) {
    fail("shader helper return values must have trivial destruction; return data or use an output reference", F->getLocation()); return {};
  }
  Frame.Closure = Object.Closure;
  if (!F->getReturnType()->isVoidType() && !F->getReturnType()->isReferenceType())
    Frame.Result = temporary(F->getReturnType(), "return.value");
  std::vector<std::pair<SpirvValue, SpirvValue>> CopyOut;
  for (unsigned I = 0; I < Args.size(); ++I) {
    const ParmVarDecl *P = F->getParamDecl(I);
    if (needsDestruction(P->getType())) {
      fail("shader parameters with nontrivial destruction must be passed by reference", P->getLocation()); return {};
    }
    const auto *Direction = P->getAttr<HLSLParamModifierAttr>();
    bool Output = Direction && (Direction->isOut() || Direction->isInOut());
    if (Args[I].Stream || Args[I].Interface || Args[I].Closure || Args[I].Resource ||
        (P->getType()->isReferenceType() && !Output) || P->getType()->isPointerType()) {
      SpirvValue Bound = Args[I];
      if (!Bound.Address && !Bound.Closure && !Bound.Resource && !Bound.Stream && !Bound.Interface) {
        Bound = temporary(P->getType().getNonReferenceType()); store(Bound, Args[I]);
      }
      Locals[P->getCanonicalDecl()] = Bound;
    } else {
      SpirvValue Parameter = temporary(P->getType(), P->getName());
      if (!Direction || !Direction->isOut() || Direction->isInOut()) store(Parameter, Args[I]);
      Locals[P->getCanonicalDecl()] = Parameter;
      if (Output) CopyOut.emplace_back(Args[I], Parameter);
    }
  }
  Calls.push_back(Frame);
  // An inline function is a single-iteration structured loop. A source
  // return exits this loop, which is a legal structured exit from nested
  // selections and loops. No recursive SPIR-V call graph is introduced.
  uint32_t Header = id(), Start = id(), Continue = id();
  branch(Header); label(Header);
  Body.emit(OpLoopMerge, {Frame.Merge, Continue, 0}); branch(Start); label(Start);
  if (const auto *Constructor = dyn_cast<CXXConstructorDecl>(F)) {
    for (const auto *Init : Constructor->inits()) {
      if (Init->isMemberInitializer()) {
        const FieldDecl *Field = Init->getMember();
        store(access(Object, {constant(integerType(), Constructor->getParent()->getNumBases() + Field->getFieldIndex())}, Field->getType()),
              expression(Init->getInit()));
      } else if (Init->isBaseInitializer()) {
        unsigned Index = 0;
        for (const auto &Base : Constructor->getParent()->bases()) {
          if (Ctx.hasSameUnqualifiedType(Base.getType(), QualType(Init->getBaseClass(), 0))) break;
          ++Index;
        }
        QualType BaseType(Init->getBaseClass(), 0);
        store(access(Object, {constant(integerType(), Index)}, BaseType), expression(Init->getInit()));
      } else if (Init->isDelegatingInitializer()) {
        store(Object, expression(Init->getInit()));
      }
    }
  }
  statement(F->getBody()); branch(Frame.Merge);
  label(Continue); branch(Header); label(Frame.Merge);
  if (const auto *Dtor = dyn_cast<CXXDestructorDecl>(F)) destroyMembers(Object, Dtor->getParent());
  for (auto [Target, Source] : CopyOut) store(Target, Source);
  Cleanups.resize(Frame.CleanupDepth);
  Calls.pop_back(); Locals = std::move(SavedLocals); FlowTargets = std::move(SavedFlow);
  return Frame.Result ? load(Frame.Result) : SpirvValue{};
}
}
