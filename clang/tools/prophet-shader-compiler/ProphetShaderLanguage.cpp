// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#include "ProphetShaderLanguage.h"
#include "clang/AST/ASTContext.h"
#include "clang/AST/RecursiveASTVisitor.h"
#include "clang/Basic/Diagnostic.h"
#include "clang/Basic/SourceManager.h"

using namespace clang;
namespace {
class HlslLanguageValidator : public RecursiveASTVisitor<HlslLanguageValidator> {
  ASTContext &Context;
  bool reject(SourceLocation Loc, llvm::StringRef Feature) {
    unsigned ID = Context.getDiagnostics().getCustomDiagID(DiagnosticsEngine::Error,
        "%0 is a Shader C++ feature and is not enabled in pure HLSL");
    Context.getDiagnostics().Report(Loc, ID) << Feature;
    return false;
  }
public:
  explicit HlslLanguageValidator(ASTContext &C) : Context(C) {}
  bool TraverseDecl(Decl *D) {
    if (!D || D->isImplicit() ||
        (D->getLocation().isValid() && Context.getSourceManager().isInSystemHeader(D->getLocation())))
      return true;
    return RecursiveASTVisitor::TraverseDecl(D);
  }
  bool VisitFunctionDecl(FunctionDecl *F) {
    return !F->isConstexpr() || reject(F->getLocation(), "constexpr function");
  }
  bool VisitVarDecl(VarDecl *V) {
    return !V->isConstexpr() || reject(V->getLocation(), "constexpr variable");
  }
  bool VisitCXXRecordDecl(CXXRecordDecl *R) {
    return !R->isUnion() || reject(R->getLocation(), "union");
  }
  bool VisitAutoTypeLoc(AutoTypeLoc T) {
    return reject(T.getBeginLoc(), "auto type deduction");
  }
  bool VisitAccessSpecDecl(AccessSpecDecl *D) {
    return reject(D->getLocation(), "class access specifier");
  }
  bool VisitIfStmt(IfStmt *S) {
    return !S->isConstexpr() || reject(S->getIfLoc(), "if constexpr");
  }
  bool VisitCXXFoldExpr(CXXFoldExpr *E) {
    return reject(E->getBeginLoc(), "fold expression");
  }
  bool VisitDesignatedInitExpr(DesignatedInitExpr *E) {
    return reject(E->getBeginLoc(), "designated initializer");
  }
  bool VisitDecltypeTypeLoc(DecltypeTypeLoc T) {
    return reject(T.getBeginLoc(), "decltype");
  }
  bool VisitLambdaExpr(LambdaExpr *E) { return reject(E->getBeginLoc(), "lambda"); }
  bool VisitDecompositionDecl(DecompositionDecl *D) {
    return reject(D->getLocation(), "structured binding");
  }
  bool VisitCXXForRangeStmt(CXXForRangeStmt *S) { return reject(S->getForLoc(), "range-for"); }
};
}
namespace prophet {
bool validateShaderLanguage(ASTContext &Context) {
  if (!Context.getLangOpts().HLSL || Context.getLangOpts().HLSLCPlusPlus) return true;
  return HlslLanguageValidator(Context).TraverseDecl(Context.getTranslationUnitDecl());
}
}
