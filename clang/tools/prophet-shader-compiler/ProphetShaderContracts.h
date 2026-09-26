// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PROPHET_SHADER_CONTRACTS_H
#define PROPHET_SHADER_CONTRACTS_H

#include "clang/AST/ASTContext.h"
#include "clang/AST/Attr.h"
#include "llvm/ADT/StringRef.h"
#include "llvm/Support/JSON.h"

namespace prophet {
const clang::AnnotateAttr *shaderAttribute(const clang::Decl *Declaration,
                                         llvm::StringRef Name);
std::string shaderAttributeString(const clang::AnnotateAttr *Attribute,
                                  unsigned Index = 0);
std::string shaderTypeId(const clang::Decl *Declaration);
clang::QualType shaderArrayShapeType(clang::ASTContext &Context, const clang::RecordDecl *Declaration,
                                     std::string &Diagnostic);
bool extractShaderContracts(clang::ASTContext &Context, llvm::StringRef ModuleId,
                            std::string &Json, std::string &Diagnostic);
llvm::json::Value shaderValueShape(clang::ASTContext &Context, clang::QualType Type,
                                   std::string &Diagnostic);
std::string shaderValueSignature(clang::ASTContext &Context, clang::QualType Type);
}
#endif
