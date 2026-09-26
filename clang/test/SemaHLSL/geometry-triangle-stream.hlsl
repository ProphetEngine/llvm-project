// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-geometry -x hlsl -finclude-default-header -hlsl-entry main -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple dxil-pc-shadermodel6.3-geometry -x hlsl -finclude-default-header -hlsl-entry main -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-geometry -x hlsl -finclude-default-header -hlsl-entry main -ast-dump %s | FileCheck %s
// expected-no-diagnostics

struct Vertex { float4 position : SV_Position; };
[maxvertexcount(3)]
void main(triangle Vertex input[3], inout TriangleStream<Vertex> output) {
  for (uint i = 0; i < 3; ++i) output.Append(input[i]);
}

// CHECK-DAG: HLSLMaxVertexCountAttr {{.*}} 3
// CHECK-DAG: HLSLGeometryInputAttr
// CHECK-DAG: HLSLAppliedSemanticAttr {{.*}} "SV_Position" 0
