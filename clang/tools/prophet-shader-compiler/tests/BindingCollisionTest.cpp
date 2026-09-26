//===-- BindingCollisionTest.cpp - Native HLSL binding ABI checks ---------===//
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception

#include "prophet_shader_compiler.h"
#include <array>
#include <cstdint>
#include <cstring>
#include <filesystem>
#include <fstream>
#include <iostream>
#include <string>
#include <string_view>
#include <system_error>

namespace {
struct Case {
  const char *Name;
  const char *Source;
  bool ExpectSuccess;
};

constexpr std::array<Case, 7> Cases = {{
    {"t0-s0-collision",
     R"(Texture2D<float4> Source : register(t0, space0);
SamplerState Linear : register(s0, space0);
float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
  return Source.SampleLevel(Linear, uv, 0.0);
})",
     false},
    {"t0-u0-collision",
     R"(Texture2D<float4> Source : register(t0, space0);
RWTexture2D<float4> Target : register(u0, space0);
float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
  float4 value = Source.Load(int3(0, 0, 0));
  Target[uint2(0, 0)] = value;
  return value;
})",
     false},
    {"b0-s0-collision",
     R"(cbuffer Params : register(b0, space0) { float4 Color; };
SamplerState Linear : register(s0, space0);
float4 main() : SV_Target0 { return Color; })",
     false},
    {"explicit-vk-collision",
     R"(Texture2D<float4> Source : register(t0, space0);
[[vk::binding(0, 0)]] SamplerState Linear : register(s0, space0);
float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
  return Source.SampleLevel(Linear, uv, 0.0);
})",
     false},
    {"explicit-vk-disambiguated",
     R"(Texture2D<float4> Source : register(t0, space0);
[[vk::binding(1, 0)]] SamplerState Linear : register(s0, space0);
float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
  return Source.SampleLevel(Linear, uv, 0.0);
})",
     true},
    {"legacy-t0-s1",
     R"(Texture2D<float4> Source : register(t0, space0);
SamplerState Linear : register(s1, space0);
float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
  return Source.SampleLevel(Linear, uv, 0.0);
})",
     true},
    {"separate-spaces",
     R"(Texture2D<float4> Source : register(t0, space0);
SamplerState Linear : register(s0, space1);
float4 main(float2 uv : TEXCOORD0) : SV_Target0 {
  return Source.SampleLevel(Linear, uv, 0.0);
})",
     true},
}};

bool runCase(const Case &C, const std::filesystem::path &Output) {
  std::string_view Source(C.Source);
  ProphetShaderCompileRequest Request = {};
  Request.struct_size = sizeof(Request);
  Request.language = PROPHET_SHADER_LANGUAGE_HLSL;
  Request.stage = PROPHET_SHADER_STAGE_FRAGMENT;
  Request.target_environment = PROPHET_SHADER_ENVIRONMENT_VULKAN_1_3;
  Request.optimization_level = 2;
  Request.source = reinterpret_cast<const uint8_t *>(Source.data());
  Request.source_size = Source.size();
  Request.source_name = C.Name;
  Request.entry_point = "main";

  ProphetShaderCompileResult *Result = nullptr;
  int32_t Status = prophet_shader_compile(&Request, &Result);
  if (!Result) {
    std::cerr << C.Name << ": missing compiler result\n";
    return false;
  }

  std::string Diagnostic;
  if (Result->diagnostics)
    Diagnostic.assign(Result->diagnostics, Result->diagnostics_size);
  bool OK = C.ExpectSuccess
                ? Status == PROPHET_SHADER_STATUS_SUCCESS &&
                      Result->spirv_size >= 20 && Result->spirv != nullptr
                : Status == PROPHET_SHADER_STATUS_UNSUPPORTED_FEATURE &&
                      Diagnostic.find("descriptor set 0 binding 0") !=
                          std::string::npos &&
                      Result->spirv_size == 0;

  if (OK && C.ExpectSuccess) {
    uint32_t Magic = 0;
    std::memcpy(&Magic, Result->spirv, sizeof(Magic));
    OK = Magic == 0x07230203;
    if (OK) {
      std::ofstream File(Output / (std::string(C.Name) + ".spv"),
                         std::ios::binary);
      File.write(reinterpret_cast<const char *>(Result->spirv),
                 Result->spirv_size);
      OK = File.good();
    }
  }

  if (!OK)
    std::cerr << C.Name << ": status=" << Status << ", diagnostic="
              << Diagnostic << '\n';
  prophet_shader_free_result(Result);
  return OK;
}
} // namespace

int main(int argc, char **argv) {
  if (argc != 2) {
    std::cerr << "usage: prophet_shader_binding_tests OUTPUT_DIRECTORY\n";
    return 2;
  }
  std::filesystem::path Output(argv[1]);
  std::error_code Error;
  std::filesystem::create_directories(Output, Error);
  if (Error) {
    std::cerr << Error.message() << '\n';
    return 2;
  }

  unsigned Passed = 0;
  for (const Case &C : Cases)
    Passed += runCase(C, Output);
  std::cout << "binding cases: " << Passed << '/' << Cases.size() << '\n';
  return Passed == Cases.size() ? 0 : 1;
}