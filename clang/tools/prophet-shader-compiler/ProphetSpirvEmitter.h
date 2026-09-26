//===-- ProphetSpirvEmitter.h - Shared shader AST lowering ----------------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
#ifndef PROPHET_SPIRV_EMITTER_H
#define PROPHET_SPIRV_EMITTER_H

#include "clang/AST/ASTContext.h"
#include "llvm/ADT/StringRef.h"
#include <cstdint>
#include <string>
#include <vector>

namespace prophet {
// Both source languages share this action after Clang parsing and Sema.
// This path does not invoke LLVM IR or target code generation.
bool emitShaderSpirv(clang::ASTContext &Context, llvm::StringRef EntryPoint,
                     uint32_t Stage, uint32_t Environment, uint32_t Flags,
                     std::vector<uint8_t> &Output, std::string &Diagnostic);
bool emitShaderObject(clang::ASTContext &Context, llvm::StringRef ModuleId,
                      llvm::StringRef EntryPoint, uint32_t Stage,
                      uint32_t Environment, uint32_t Flags,
                      std::vector<uint8_t> &Output, std::string &Manifest,
                      std::string &Diagnostic);
}
#endif
