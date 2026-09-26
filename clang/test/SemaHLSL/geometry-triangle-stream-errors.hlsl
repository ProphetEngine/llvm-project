// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-geometry -x hlsl -finclude-default-header -hlsl-entry main -fsyntax-only -verify -DMISSING_MAX %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-geometry -x hlsl -finclude-default-header -hlsl-entry main -fsyntax-only -verify -DBAD_COUNT %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-geometry -x hlsl -finclude-default-header -hlsl-entry main -fsyntax-only -verify -DBAD_INPUT %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-geometry -x hlsl -finclude-default-header -hlsl-entry main -fsyntax-only -verify -DBAD_STREAM %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-geometry -x hlsl -finclude-default-header -hlsl-entry main -fsyntax-only -verify -DBAD_MODIFIER %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-vertex -x hlsl -finclude-default-header -hlsl-entry main -fsyntax-only -verify -DBAD_STAGE %s

struct Vertex { float4 position : SV_Position; };

#if defined(MISSING_MAX)
void main(triangle Vertex input[3], inout TriangleStream<Vertex> output) { // expected-error {{geometry shader entry requires a maxvertexcount attribute}}
  output.Append(input[0]);
}
#elif defined(BAD_COUNT)
[maxvertexcount(0)] // expected-error {{maxvertexcount must be between 1 and 1024}}
void main(triangle Vertex input[3], inout TriangleStream<Vertex> output) { // expected-error {{geometry shader entry requires a maxvertexcount attribute}}
  output.Append(input[0]);
}
#elif defined(BAD_INPUT)
[maxvertexcount(3)]
void main(triangle Vertex input[2], inout TriangleStream<Vertex> output) { // expected-error {{triangle input must be an array of exactly three records}}
  output.Append(input[0]);
}
#elif defined(BAD_STREAM)
[maxvertexcount(3)]
void main(triangle Vertex input[3], inout TriangleStream<uint> output) { // expected-error {{geometry output must be an inout TriangleStream of a record type}}
  output.Append(1u);
}
#elif defined(BAD_MODIFIER)
[maxvertexcount(3)]
void main(triangle Vertex input[3], TriangleStream<Vertex> output) { // expected-error {{geometry output must be an inout TriangleStream of a record type}}
  output.Append(input[0]);
}
#elif defined(BAD_STAGE)
[maxvertexcount(3)] // expected-error {{attribute 'maxvertexcount' is unsupported in 'vertex' shaders}}
float4 main(uint vertexId : SV_VertexID) : SV_Position {
  return float4(float(vertexId), 0.0f, 0.0f, 1.0f);
}
#endif
