// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PROPHET_SHADER_LANGUAGE_H
#define PROPHET_SHADER_LANGUAGE_H
namespace clang { class ASTContext; }
namespace prophet { bool validateShaderLanguage(clang::ASTContext &Context); }
#endif
