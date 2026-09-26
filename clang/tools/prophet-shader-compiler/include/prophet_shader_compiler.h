//===-- prophet_shader_compiler.h - Stable shader compiler C ABI ---------===//
//
// Part of the ProphetEngine LLVM fork, under the Apache License v2.0 with
// LLVM Exceptions. See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//

#ifndef PROPHET_SHADER_COMPILER_H
#define PROPHET_SHADER_COMPILER_H

#include <stddef.h>
#include <stdint.h>

#if defined(_WIN32)
#  if defined(PROPHET_SHADER_COMPILER_BUILD)
#    define PROPHET_SHADER_API __declspec(dllexport)
#  else
#    define PROPHET_SHADER_API __declspec(dllimport)
#  endif
#else
#  define PROPHET_SHADER_API __attribute__((visibility("default")))
#endif

#ifdef __cplusplus
extern "C" {
#endif

#define PROPHET_SHADER_ABI_VERSION 1u

// Source contract v2: Shader C++ and pure HLSL share HLSL shader types.
#define PROPHET_SHADER_LANGUAGE_CONTRACT_VERSION 2u

typedef enum ProphetShaderCompileFlags {
  // register(uN,tN,bN,sN) -> binding(4*N + 0,1,2,3), preserving space.
  // Explicit vk::binding and Prophet binding annotations bypass this mapping.
  PROPHET_SHADER_SEPARATE_REGISTER_CLASSES = 1u,
  // Enable native half and the explicit float16_t/int16_t/uint16_t families.
  PROPHET_SHADER_ENABLE_16_BIT_TYPES = 2u
} ProphetShaderCompileFlags;

typedef enum ProphetShaderLanguage {
  PROPHET_SHADER_LANGUAGE_CXX = 1,
  PROPHET_SHADER_LANGUAGE_HLSL = 2
} ProphetShaderLanguage;

typedef enum ProphetShaderStage {
  PROPHET_SHADER_STAGE_AUTOMATIC = 0,
  PROPHET_SHADER_STAGE_COMPUTE = 1,
  PROPHET_SHADER_STAGE_VERTEX = 2,
  PROPHET_SHADER_STAGE_FRAGMENT = 3,
  PROPHET_SHADER_STAGE_GEOMETRY = 4,
  PROPHET_SHADER_STAGE_HULL = 5,
  PROPHET_SHADER_STAGE_DOMAIN = 6,
  PROPHET_SHADER_STAGE_TASK = 7,
  PROPHET_SHADER_STAGE_MESH = 8,
  PROPHET_SHADER_STAGE_LIBRARY = 9,
  PROPHET_SHADER_STAGE_RAY_GENERATION = 10,
  PROPHET_SHADER_STAGE_INTERSECTION = 11,
  PROPHET_SHADER_STAGE_ANY_HIT = 12,
  PROPHET_SHADER_STAGE_CLOSEST_HIT = 13,
  PROPHET_SHADER_STAGE_MISS = 14,
  PROPHET_SHADER_STAGE_CALLABLE = 15
} ProphetShaderStage;

typedef enum ProphetShaderTargetEnvironment {
  PROPHET_SHADER_ENVIRONMENT_AUTOMATIC = 0,
  PROPHET_SHADER_ENVIRONMENT_VULKAN_1_1 = 1,
  PROPHET_SHADER_ENVIRONMENT_VULKAN_1_2 = 2,
  PROPHET_SHADER_ENVIRONMENT_VULKAN_1_3 = 3
} ProphetShaderTargetEnvironment;

typedef enum ProphetShaderStatus {
  PROPHET_SHADER_STATUS_SUCCESS = 0,
  PROPHET_SHADER_STATUS_INVALID_ARGUMENT = 1,
  PROPHET_SHADER_STATUS_COMPILATION_ERROR = 2,
  PROPHET_SHADER_STATUS_UNSUPPORTED_FEATURE = 3,
  PROPHET_SHADER_STATUS_INTERNAL_ERROR = 4
} ProphetShaderStatus;

// All strings are UTF-8, NUL terminated. `source` may contain NUL bytes.
// The caller owns every pointer in this request for the duration of the call.
// ABI v1 requires struct_size == sizeof(ProphetShaderCompileRequest).
// `extra_args` accepts include/define/warning/language options only; target,
// stage, input and output are fixed by the request and cannot be overridden.
typedef struct ProphetShaderCompileRequest {
  uint32_t struct_size;
  uint32_t language;
  uint32_t stage;
  uint32_t target_environment;
  uint32_t optimization_level; // 0..3
  uint32_t flags;              // ProphetShaderCompileFlags; unknown bits reject.
  const uint8_t *source;
  size_t source_size;
  const char *source_name;
  const char *entry_point;
  const char *const *extra_args;
  size_t extra_arg_count;
} ProphetShaderCompileRequest;

// The result and both payloads are owned by the DLL. Only
// prophet_shader_free_result may release them. `spirv` is populated only on
// success. `diagnostics` is NUL terminated for convenience; its size excludes
// that terminator. The returned SPIR-V bytes are a complete binary module.
typedef struct ProphetShaderCompileResult {
  uint32_t struct_size;
  uint32_t status;
  uint8_t *spirv;
  size_t spirv_size;
  char *diagnostics;
  size_t diagnostics_size;
} ProphetShaderCompileResult;

PROPHET_SHADER_API uint32_t prophet_shader_abi_version(void);
// Language / SDK semantics are versioned independently of the stable C layout.
PROPHET_SHADER_API uint32_t prophet_shader_language_contract_version(void);

// Always initializes *result when `result` itself is non-null. Returns the
// same status written into the result. No exception crosses this C boundary.
PROPHET_SHADER_API int32_t prophet_shader_compile(
    const ProphetShaderCompileRequest *request,
    ProphetShaderCompileResult **result);

PROPHET_SHADER_API void
prophet_shader_free_result(ProphetShaderCompileResult *result);

// Module ABI is additive: the original entry ABI stays binary compatible.
#define PROPHET_SHADER_MODULE_ABI_VERSION 1u
typedef enum ProphetShaderModuleOperation {
  PROPHET_SHADER_EXTRACT_CONTRACTS = 1,
  PROPHET_SHADER_COMPILE_OBJECT = 2
} ProphetShaderModuleOperation;

typedef struct ProphetShaderModuleRequest {
  uint32_t struct_size;
  uint32_t operation;
  ProphetShaderCompileRequest compile;
  const char *module_id;
} ProphetShaderModuleRequest;

typedef struct ProphetShaderModuleResult {
  uint32_t struct_size;
  uint32_t status;
  uint8_t *spirv;
  size_t spirv_size;
  char *metadata;
  size_t metadata_size;
  char *diagnostics;
  size_t diagnostics_size;
} ProphetShaderModuleResult;

PROPHET_SHADER_API uint32_t prophet_shader_module_abi_version(void);
PROPHET_SHADER_API int32_t prophet_shader_compile_module(
    const ProphetShaderModuleRequest *request, ProphetShaderModuleResult **result);
PROPHET_SHADER_API void prophet_shader_free_module_result(ProphetShaderModuleResult *result);

#ifdef __cplusplus
}
#endif

#endif // PROPHET_SHADER_COMPILER_H
