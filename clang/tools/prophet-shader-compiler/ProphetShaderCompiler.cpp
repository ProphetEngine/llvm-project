//===-- ProphetShaderCompiler.cpp - In-process Clang shader DLL -----------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "ProphetSpirvEmitter.h"
#include "ProphetShaderLanguage.h"
#include "ProphetShaderContracts.h"
#include "prophet_shader_compiler.h"
#include "clang/AST/ASTConsumer.h"
#include "clang/AST/Decl.h"
#include "clang/AST/DeclTemplate.h"
#include "clang/Frontend/CompilerInstance.h"
#include "clang/Frontend/CompilerInvocation.h"
#include "clang/Frontend/FrontendAction.h"
#include "clang/Frontend/FrontendActions.h"
#include "clang/Frontend/TextDiagnosticPrinter.h"
#include "clang/Lex/PreprocessorOptions.h"
#include "clang/Lex/HeaderSearchOptions.h"
#include "clang/Options/OptionUtils.h"
#include "llvm/ADT/SmallVector.h"
#include "llvm/Support/MemoryBuffer.h"
#include "llvm/Support/FileSystem.h"
#include "llvm/Support/Path.h"
#include "llvm/Support/VirtualFileSystem.h"
#include <cstdlib>
#include <cstring>
#include <exception>
#include <map>
#include <memory>
#include <mutex>
#include <string>
#include <vector>
#if defined(_WIN32)
#define NOMINMAX
#include <windows.h>
#endif

using namespace clang;
using namespace llvm;

namespace {

std::mutex &compileMutex() {
  static std::mutex Mutex;
  return Mutex;
}

std::string resourceDirectory() {
#if defined(_WIN32)
  // llvm::sys::fs::getMainExecutable ignores MainExecAddr on Windows and
  // resolves the host .exe. Managed callers load us as a DLL instead.
  HMODULE Module = nullptr;
  if (!GetModuleHandleExW(GET_MODULE_HANDLE_EX_FLAG_FROM_ADDRESS |
                              GET_MODULE_HANDLE_EX_FLAG_UNCHANGED_REFCOUNT,
                          reinterpret_cast<LPCWSTR>(&prophet_shader_compile),
                          &Module))
    return {};
  std::vector<wchar_t> Path(32768);
  DWORD Length = GetModuleFileNameW(Module, Path.data(), DWORD(Path.size()));
  if (!Length || Length >= Path.size())
    return {};
  int Bytes = WideCharToMultiByte(CP_UTF8, 0, Path.data(), int(Length),
                                  nullptr, 0, nullptr, nullptr);
  if (Bytes <= 0)
    return {};
  std::string Utf8Path(size_t(Bytes), '\0');
  if (!WideCharToMultiByte(CP_UTF8, 0, Path.data(), int(Length),
                           Utf8Path.data(), Bytes, nullptr, nullptr))
    return {};
  return GetResourcesPath(Utf8Path);
#else
  return GetResourcesPath("prophet_shader_compiler",
                          reinterpret_cast<void *>(&prophet_shader_compile));
#endif
}

bool stageName(uint32_t Stage, StringRef &Name) {
  switch (Stage) {
  case PROPHET_SHADER_STAGE_COMPUTE: Name = "compute"; return true;
  case PROPHET_SHADER_STAGE_VERTEX: Name = "vertex"; return true;
  case PROPHET_SHADER_STAGE_FRAGMENT: Name = "pixel"; return true;
  case PROPHET_SHADER_STAGE_GEOMETRY: Name = "geometry"; return true;
  case PROPHET_SHADER_STAGE_HULL: Name = "hull"; return true;
  case PROPHET_SHADER_STAGE_DOMAIN: Name = "domain"; return true;
  case PROPHET_SHADER_STAGE_TASK: Name = "amplification"; return true;
  case PROPHET_SHADER_STAGE_MESH: Name = "mesh"; return true;
  case PROPHET_SHADER_STAGE_LIBRARY: Name = "library"; return true;
  case PROPHET_SHADER_STAGE_RAY_GENERATION: Name = "raygeneration"; return true;
  case PROPHET_SHADER_STAGE_INTERSECTION: Name = "intersection"; return true;
  case PROPHET_SHADER_STAGE_ANY_HIT: Name = "anyhit"; return true;
  case PROPHET_SHADER_STAGE_CLOSEST_HIT: Name = "closesthit"; return true;
  case PROPHET_SHADER_STAGE_MISS: Name = "miss"; return true;
  case PROPHET_SHADER_STAGE_CALLABLE: Name = "callable"; return true;
  default: return false;
  }
}

bool environmentName(uint32_t Environment, StringRef &Name) {
  switch (Environment) {
  case PROPHET_SHADER_ENVIRONMENT_AUTOMATIC:
  case PROPHET_SHADER_ENVIRONMENT_VULKAN_1_3:
    Name = "vulkan1.3"; return true;
  case PROPHET_SHADER_ENVIRONMENT_VULKAN_1_1:
    Name = "vulkan1.1"; return true;
  case PROPHET_SHADER_ENVIRONMENT_VULKAN_1_2:
    Name = "vulkan1.2"; return true;
  default:
    return false;
  }
}

bool isAllowedExtraArg(StringRef Arg, uint32_t Language) {
  // The DLL owns target/output selection. An argument array is passed to
  // Clang without a shell, but target-changing cc1 switches would invalidate
  // the requested binding and stage contract.
  if (Arg.starts_with("-D") && Arg.size() > 2)
    return true;
  if (Arg.starts_with("-U") && Arg.size() > 2)
    return true;
  if (Arg.starts_with("-I") && Arg.size() > 2)
    return true;
  if (Arg.starts_with("-isystem=") && Arg.size() > 9)
    return true;
  if (Arg.starts_with("-W") && Arg.size() > 2)
    return true;
  if (Language == PROPHET_SHADER_LANGUAGE_CXX &&
      (Arg == "-std=c++20" || Arg == "-std=c++23" || Arg == "-std=c++26"))
    return true;
  if (Language == PROPHET_SHADER_LANGUAGE_HLSL &&
      (Arg == "-std=hlsl202x" || Arg == "-std=hlsl2021"))
    return true;
  return false;
}

ProphetShaderCompileResult *makeResult(uint32_t Status, StringRef Diagnostic,
                                       ArrayRef<uint8_t> Spirv = {}) {
  auto *Result = static_cast<ProphetShaderCompileResult *>(
      std::calloc(1, sizeof(ProphetShaderCompileResult)));
  if (!Result)
    return nullptr;
  Result->struct_size = sizeof(ProphetShaderCompileResult);
  Result->status = Status;
  Result->diagnostics_size = Diagnostic.size();
  Result->diagnostics = static_cast<char *>(std::malloc(Diagnostic.size() + 1));
  if (!Result->diagnostics) {
    std::free(Result);
    return nullptr;
  }
  std::memcpy(Result->diagnostics, Diagnostic.data(), Diagnostic.size());
  Result->diagnostics[Diagnostic.size()] = '\0';
  if (!Spirv.empty()) {
    Result->spirv = static_cast<uint8_t *>(std::malloc(Spirv.size()));
    if (!Result->spirv) {
      std::free(Result->diagnostics);
      std::free(Result);
      return nullptr;
    }
    std::memcpy(Result->spirv, Spirv.data(), Spirv.size());
    Result->spirv_size = Spirv.size();
  }
  return Result;
}

class ShaderConsumer final : public ASTConsumer {
  StringRef Entry;
  uint32_t Stage, Environment, Flags;
  std::vector<uint8_t> &Output;
  std::string &Error;
  bool &Succeeded;
  uint32_t Operation;
  StringRef ModuleId;
  std::string *Metadata;

public:
  ShaderConsumer(StringRef Entry, uint32_t Stage, uint32_t Environment, uint32_t Flags,
                 std::vector<uint8_t> &Output, std::string &Error,
                 bool &Succeeded, uint32_t Operation, StringRef ModuleId,
                 std::string *Metadata)
      : Entry(Entry), Stage(Stage), Environment(Environment), Flags(Flags), Output(Output),
        Error(Error), Succeeded(Succeeded), Operation(Operation), ModuleId(ModuleId), Metadata(Metadata) {}

  bool shouldSkipFunctionBody(Decl *D) override {
    const auto *F = dyn_cast<FunctionDecl>(D);
    return !F || !F->isConstexpr();
  }

  void HandleTranslationUnit(ASTContext &Context) override {
    if (Context.getDiagnostics().hasErrorOccurred() || !prophet::validateShaderLanguage(Context)) return;
    if (Operation == PROPHET_SHADER_EXTRACT_CONTRACTS) {
      Succeeded = prophet::extractShaderContracts(Context, ModuleId, *Metadata, Error);
      return;
    }
    if (Operation == PROPHET_SHADER_COMPILE_OBJECT) {
      std::string ContractJson;
      if (!prophet::extractShaderContracts(Context, ModuleId, ContractJson, Error)) return;
      Succeeded = prophet::emitShaderObject(Context, ModuleId, Entry, Stage, Environment, Flags,
                                            Output, *Metadata, Error);
      if (Succeeded) {
        auto Package = json::parse(ContractJson);
        auto Manifest = json::parse(*Metadata);
        if (!Package || !Manifest || !Manifest->getAsObject()) {
          if (!Package) consumeError(Package.takeError());
          if (!Manifest) consumeError(Manifest.takeError());
          Succeeded = false; Error = "native module metadata is invalid"; return;
        }
        (*Manifest->getAsObject())["contractPackage"] = std::move(*Package);
        Metadata->clear();
        raw_string_ostream Stream(*Metadata); Stream << *Manifest;
      }
      return;
    }
    Succeeded = prophet::emitShaderSpirv(Context, Entry, Stage, Environment, Flags,
                                        Output, Error);
  }
};
class ShaderAction final : public ASTFrontendAction {
  StringRef Entry;
  uint32_t Stage, Environment, Flags;
  std::vector<uint8_t> &Output;
  std::string &Error;
  bool &Succeeded;
  uint32_t Operation;
  StringRef ModuleId;
  std::string *Metadata;

public:
  ShaderAction(StringRef Entry, uint32_t Stage, uint32_t Environment, uint32_t Flags,
               std::vector<uint8_t> &Output, std::string &Error,
               bool &Succeeded, uint32_t Operation, StringRef ModuleId,
               std::string *Metadata)
      : Entry(Entry), Stage(Stage), Environment(Environment), Flags(Flags), Output(Output),
        Error(Error), Succeeded(Succeeded), Operation(Operation), ModuleId(ModuleId), Metadata(Metadata) {}

  std::unique_ptr<ASTConsumer> CreateASTConsumer(CompilerInstance &, StringRef) override {
    return std::make_unique<ShaderConsumer>(Entry, Stage, Environment, Flags, Output,
                                            Error, Succeeded, Operation, ModuleId, Metadata);
  }
};
uint32_t compileOne(const ProphetShaderCompileRequest &Request,
                    std::vector<uint8_t> &Output, std::string &Diagnostic,
                    uint32_t Operation = 0, StringRef ModuleId = {},
                    std::string *Metadata = nullptr) {
  uint32_t EffectiveStage = Request.stage == PROPHET_SHADER_STAGE_AUTOMATIC
                                ? (Operation ? PROPHET_SHADER_STAGE_LIBRARY : Request.language == PROPHET_SHADER_LANGUAGE_CXX
                                       ? PROPHET_SHADER_STAGE_COMPUTE
                                       : PROPHET_SHADER_STAGE_LIBRARY)
                                : Request.stage;
  if (Operation && Request.entry_point && !*Request.entry_point)
    EffectiveStage = PROPHET_SHADER_STAGE_LIBRARY;
  StringRef Stage;
  StringRef Environment;
  if (!stageName(EffectiveStage, Stage) ||
      !environmentName(Request.target_environment, Environment)) {
    Diagnostic = "invalid shader stage or target environment";
    return PROPHET_SHADER_STATUS_INVALID_ARGUMENT;
  }
  if (Request.language != PROPHET_SHADER_LANGUAGE_CXX &&
      Request.language != PROPHET_SHADER_LANGUAGE_HLSL) {
    Diagnostic = "invalid shader language";
    return PROPHET_SHADER_STATUS_INVALID_ARGUMENT;
  }
  if (!Operation && Request.target_environment == PROPHET_SHADER_ENVIRONMENT_VULKAN_1_1 &&
      (EffectiveStage == PROPHET_SHADER_STAGE_TASK ||
       EffectiveStage == PROPHET_SHADER_STAGE_MESH || EffectiveStage >= 9)) {
    Diagnostic = "task, mesh and ray shader binaries require Vulkan 1.2 or later";
    return PROPHET_SHADER_STATUS_UNSUPPORTED_FEATURE;
  }
  if (Request.optimization_level > 3 || (Request.flags & ~(PROPHET_SHADER_SEPARATE_REGISTER_CLASSES | PROPHET_SHADER_ENABLE_16_BIT_TYPES)) != 0) {
    Diagnostic = "invalid optimization level or unknown compiler flags";
    return PROPHET_SHADER_STATUS_INVALID_ARGUMENT;
  }
  if (!Request.source || !Request.entry_point ||
      (EffectiveStage != PROPHET_SHADER_STAGE_LIBRARY &&
       !*Request.entry_point) ||
      (Request.extra_arg_count && !Request.extra_args)) {
    Diagnostic = "source, entry_point and extra_args must be valid";
    return PROPHET_SHADER_STATUS_INVALID_ARGUMENT;
  }

  StringRef Source(reinterpret_cast<const char *>(Request.source),
                   Request.source_size);
  const char *DefaultName =
      Request.language == PROPHET_SHADER_LANGUAGE_HLSL ? "shader.hlsl"
                                                       : "shader.cpp";
  std::string SourceName =
      Request.source_name && *Request.source_name ? Request.source_name
                                                   : DefaultName;
  // The frontend target supplies shader language types; binary emission
  // and the requested SPIR-V version are owned by the shared AST emitter.
  StringRef FrontendEnvironment =
      Request.language == PROPHET_SHADER_LANGUAGE_CXX || Operation
          ? StringRef("vulkan1.3")
          : Environment;
  std::string Triple = ("spirv-unknown-" + FrontendEnvironment + "-" + Stage).str();
  std::vector<std::string> Args = {
      "-triple", Triple, "-x",
      Request.language == PROPHET_SHADER_LANGUAGE_HLSL ? "hlsl" : "c++",
      Request.language == PROPHET_SHADER_LANGUAGE_HLSL ? "-std=hlsl202x"
                                                       : "-std=c++23",
      "-O" + std::to_string(Request.optimization_level)};
  if (Request.language == PROPHET_SHADER_LANGUAGE_HLSL) {
    Args.push_back("-finclude-default-header");
    Args.push_back("-Werror=hlsl-extensions");
    Args.push_back("-Werror=hlsl-202y-extensions");
    Args.push_back("-Werror=c++14-extensions");
    Args.push_back("-Werror=c++17-extensions");
    Args.push_back("-Werror=c++20-extensions");
    Args.push_back("-Werror=c++23-extensions");
    if (EffectiveStage != PROPHET_SHADER_STAGE_LIBRARY) {
      Args.push_back("-hlsl-entry");
      Args.emplace_back(Request.entry_point);
    }
  }
  if (Request.language == PROPHET_SHADER_LANGUAGE_CXX) {
    Args.push_back("-fshader-c++");
    Args.push_back("-Wno-hlsl-extensions");
    Args.push_back("-Wno-hlsl-202y-extensions");
    Args.push_back("-ffreestanding");
    Args.push_back("-fenable-matrix");
  }
  if (Request.flags & PROPHET_SHADER_ENABLE_16_BIT_TYPES) {
    Args.push_back("-fnative-half-type");
    Args.push_back("-fnative-int16-type");
  }
  Args.push_back("-fsyntax-only");
  for (size_t I = 0; I < Request.extra_arg_count; ++I) {
    if (!Request.extra_args[I]) {
      Diagnostic = "extra_args contains a null element";
      return PROPHET_SHADER_STATUS_INVALID_ARGUMENT;
    }
    StringRef Arg(Request.extra_args[I]);
    if (!isAllowedExtraArg(Arg, Request.language)) {
      Diagnostic = ("disallowed shader compiler argument: " + Arg).str();
      return PROPHET_SHADER_STATUS_INVALID_ARGUMENT;
    }
    Args.emplace_back(Arg);
  }
  Args.push_back(SourceName);
  std::vector<const char *> Argv;
  Argv.reserve(Args.size());
  for (const std::string &Arg : Args)
    Argv.push_back(Arg.c_str());

  DiagnosticOptions ParseOptions;
  std::string ClangDiagnostics;
  raw_string_ostream DiagnosticStream(ClangDiagnostics);
  TextDiagnosticPrinter ParsePrinter(DiagnosticStream, ParseOptions);
  auto DiagnosticIDs = clang::DiagnosticIDs::create();
  DiagnosticsEngine ParseDiagnostics(DiagnosticIDs, ParseOptions,
                                     &ParsePrinter, false);
  auto Invocation = std::make_shared<CompilerInvocation>();
  bool Parsed = CompilerInvocation::CreateFromArgs(
      *Invocation, Argv, ParseDiagnostics, "prophet_shader_compiler");
  if (!Parsed) {
    DiagnosticStream.flush();
    Diagnostic = ClangDiagnostics;
    return PROPHET_SHADER_STATUS_INVALID_ARGUMENT;
  }
  if (Operation == PROPHET_SHADER_EXTRACT_CONTRACTS)
    Invocation->getFrontendOpts().SkipFunctionBodies = true;
  if (Request.language == PROPHET_SHADER_LANGUAGE_CXX) {
    // The AST emitter selects the requested entry and validates legacy SDK
    // annotations as well as native HLSL attributes.
    Invocation->getTargetOpts().HLSLEntry.clear();
  }
  if (Invocation->getHeaderSearchOpts().ResourceDir.empty()) {
    std::string ResourceDir = resourceDirectory();
    SmallString<256> HeaderPath(ResourceDir);
    sys::path::append(HeaderPath, "include", "hlsl.h");
    if (ResourceDir.empty() || !sys::fs::exists(HeaderPath)) {
      Diagnostic = "Clang HLSL resource header hlsl.h is missing beside the "
                   "shader compiler DLL";
      return PROPHET_SHADER_STATUS_INTERNAL_ERROR;
    }
    Invocation->getHeaderSearchOpts().ResourceDir = std::move(ResourceDir);
  }

  if (Request.language == PROPHET_SHADER_LANGUAGE_CXX) {
    SmallString<256> SdkInclude(resourceDirectory());
    for (unsigned I = 0; I != 3; ++I)
      SdkInclude = sys::path::parent_path(SdkInclude);
    sys::path::append(SdkInclude, "include");
    if (sys::fs::exists(SdkInclude))
      Invocation->getHeaderSearchOpts().AddPath(
          SdkInclude, frontend::Angled, false, false);
  }
  Invocation->getPreprocessorOpts().addRemappedFile(
      SourceName, MemoryBuffer::getMemBufferCopy(Source, SourceName).release());
  CompilerInstance Compiler(std::move(Invocation));
  Compiler.setVirtualFileSystem(vfs::getRealFileSystem());
  Compiler.createDiagnostics(
      new TextDiagnosticPrinter(DiagnosticStream, Compiler.getDiagnosticOpts()),
      true);

  bool ActionSucceeded = false;
  std::string LoweringError;
  auto Action = std::make_unique<ShaderAction>(Request.entry_point,
      Operation && !*Request.entry_point ? Request.stage : EffectiveStage,
      Request.target_environment, Request.flags, Output, LoweringError,
      ActionSucceeded, Operation, ModuleId, Metadata);
  bool ParsedSource;
  if (Compiler.getLangOpts().HLSL) {
    HLSLFrontendAction Wrapped(std::move(Action));
    ParsedSource = Compiler.ExecuteAction(Wrapped);
  } else {
    ParsedSource = Compiler.ExecuteAction(*Action);
  }
  ActionSucceeded = ParsedSource && ActionSucceeded;
  DiagnosticStream.flush();
  Diagnostic = ClangDiagnostics;
  if (!LoweringError.empty()) {
    if (!Diagnostic.empty() && Diagnostic.back() != '\n')
      Diagnostic += '\n';
    Diagnostic += LoweringError;
    Diagnostic += '\n';
  }
  if (Compiler.getDiagnostics().hasErrorOccurred())
    return PROPHET_SHADER_STATUS_COMPILATION_ERROR;
  if (!ActionSucceeded)
    return LoweringError.empty() ? PROPHET_SHADER_STATUS_COMPILATION_ERROR
                                 : PROPHET_SHADER_STATUS_UNSUPPORTED_FEATURE;
  if (Operation == PROPHET_SHADER_EXTRACT_CONTRACTS) return PROPHET_SHADER_STATUS_SUCCESS;
  if (Output.size() < 20 || Output.size() % 4 != 0 ||
      Output[0] != 0x03 || Output[1] != 0x02 ||
      Output[2] != 0x23 || Output[3] != 0x07) {
    Diagnostic += "Clang backend did not produce a SPIR-V binary module\n";
    Output.clear();
    return PROPHET_SHADER_STATUS_COMPILATION_ERROR;
  }
  return PROPHET_SHADER_STATUS_SUCCESS;
}

} // namespace

extern "C" uint32_t prophet_shader_abi_version(void) {
  return PROPHET_SHADER_ABI_VERSION;
}

extern "C" uint32_t prophet_shader_language_contract_version(void) {
  return PROPHET_SHADER_LANGUAGE_CONTRACT_VERSION;
}

extern "C" int32_t
prophet_shader_compile(const ProphetShaderCompileRequest *Request,
                       ProphetShaderCompileResult **Result) {
  if (!Result)
    return PROPHET_SHADER_STATUS_INVALID_ARGUMENT;
  *Result = nullptr;
#if defined(__cpp_exceptions)
  try {
#endif
    if (!Request ||
        Request->struct_size != sizeof(ProphetShaderCompileRequest)) {
      *Result = makeResult(PROPHET_SHADER_STATUS_INVALID_ARGUMENT,
                           "invalid ABI v1 request size");
      return *Result ? PROPHET_SHADER_STATUS_INVALID_ARGUMENT
                     : PROPHET_SHADER_STATUS_INTERNAL_ERROR;
    }
    // Clang's frontend and target registration use process-global state.
    // Serialize compilation until all paths are proven reentrant.
    std::lock_guard<std::mutex> Guard(compileMutex());
    std::vector<uint8_t> Spirv;
    std::string Diagnostic;
    uint32_t Status = compileOne(*Request, Spirv, Diagnostic);
    *Result = makeResult(Status, Diagnostic,
                         Status == PROPHET_SHADER_STATUS_SUCCESS
                             ? ArrayRef<uint8_t>(Spirv)
                             : ArrayRef<uint8_t>());
    return *Result ? Status : PROPHET_SHADER_STATUS_INTERNAL_ERROR;
#if defined(__cpp_exceptions)
  } catch (const std::exception &E) {
    *Result = makeResult(PROPHET_SHADER_STATUS_INTERNAL_ERROR, E.what());
    return PROPHET_SHADER_STATUS_INTERNAL_ERROR;
  } catch (...) {
    *Result = makeResult(PROPHET_SHADER_STATUS_INTERNAL_ERROR,
                         "unexpected native compiler exception");
    return PROPHET_SHADER_STATUS_INTERNAL_ERROR;
  }
#endif
}

extern "C" void
prophet_shader_free_result(ProphetShaderCompileResult *Result) {
  if (!Result)
    return;
  std::free(Result->spirv);
  std::free(Result->diagnostics);
  std::free(Result);
}

extern "C" uint32_t prophet_shader_module_abi_version(void) {
  return PROPHET_SHADER_MODULE_ABI_VERSION;
}

extern "C" void prophet_shader_free_module_result(ProphetShaderModuleResult *Result) {
  if (!Result) return;
  std::free(Result->spirv);
  std::free(Result->metadata);
  std::free(Result->diagnostics);
  std::free(Result);
}

extern "C" int32_t prophet_shader_compile_module(const ProphetShaderModuleRequest *Request,
                                                 ProphetShaderModuleResult **Result) {
  if (!Result) return PROPHET_SHADER_STATUS_INVALID_ARGUMENT;
  *Result = nullptr;
  std::vector<uint8_t> Bytes;
  std::string Diagnostic, Metadata;
  uint32_t Status = PROPHET_SHADER_STATUS_INVALID_ARGUMENT;
#if defined(__cpp_exceptions)
  try {
#endif
    if (!Request || Request->struct_size != sizeof(*Request) ||
        Request->compile.struct_size != sizeof(Request->compile) ||
        !Request->module_id || !*Request->module_id ||
        (Request->operation != PROPHET_SHADER_EXTRACT_CONTRACTS && Request->operation != PROPHET_SHADER_COMPILE_OBJECT)) {
      Diagnostic = "invalid module ABI v1 request";
    } else {
      std::lock_guard<std::mutex> Guard(compileMutex());
      Status = compileOne(Request->compile, Bytes, Diagnostic, Request->operation, Request->module_id, &Metadata);
    }
#if defined(__cpp_exceptions)
  } catch (const std::exception &E) {
    Status = PROPHET_SHADER_STATUS_INTERNAL_ERROR; Diagnostic = E.what();
  } catch (...) {
    Status = PROPHET_SHADER_STATUS_INTERNAL_ERROR; Diagnostic = "unexpected native module compiler exception";
  }
#endif
  auto *R = static_cast<ProphetShaderModuleResult *>(std::calloc(1, sizeof(ProphetShaderModuleResult)));
  if (!R) return PROPHET_SHADER_STATUS_INTERNAL_ERROR;
  R->struct_size = sizeof(*R); R->status = Status;
  auto CopyString = [](StringRef Text) -> char * {
    auto *P = static_cast<char *>(std::malloc(Text.size() + 1));
    if (P) { std::memcpy(P, Text.data(), Text.size()); P[Text.size()] = '\0'; }
    return P;
  };
  R->diagnostics = CopyString(Diagnostic); R->diagnostics_size = Diagnostic.size();
  if (Status == PROPHET_SHADER_STATUS_SUCCESS) {
    R->metadata = CopyString(Metadata); R->metadata_size = Metadata.size();
    if (!Bytes.empty()) {
      R->spirv = static_cast<uint8_t *>(std::malloc(Bytes.size())); R->spirv_size = Bytes.size();
      if (R->spirv) std::memcpy(R->spirv, Bytes.data(), Bytes.size());
    }
  }
  if (!R->diagnostics || (Status == PROPHET_SHADER_STATUS_SUCCESS &&
      (!R->metadata || (!Bytes.empty() && !R->spirv)))) {
    prophet_shader_free_module_result(R);
    return PROPHET_SHADER_STATUS_INTERNAL_ERROR;
  }
  *Result = R;
  return Status;
}
