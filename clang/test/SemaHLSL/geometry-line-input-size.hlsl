// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-geometry -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify %s

struct Vertex { float4 position : SV_Position; };
[maxvertexcount(2)]
void main(line Vertex input[3], inout LineStream<Vertex> output) { // expected-error {{line input must be an array of exactly 2 records}}
  output.Append(input[0]);
}
