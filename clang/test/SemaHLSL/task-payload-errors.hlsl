// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-compute -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify -DBAD_STAGE %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-amplification -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify -DBAD_PAYLOAD %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-mesh -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify -DBAD_MESH_INPUT %s

#if defined(BAD_STAGE)
struct Payload { uint marker; };
groupshared Payload payload;
[numthreads(1,1,1)]
void main() {
  DispatchMesh(1, 1, 1, payload); // expected-error{{dispatch mesh is only available in an amplification shader}}
}
#elif defined(BAD_PAYLOAD)
struct Payload { uint marker; };
Payload payload;
[numthreads(1,1,1)]
void main() {
  DispatchMesh(1, 1, 1, payload); // expected-error{{dispatch mesh requires a global groupshared record payload}}
}
#else
struct MeshVertex { float4 position : SV_Position; };
[outputtopology("triangle")]
[numthreads(1,1,1)]
void main(in payload uint marker, // expected-error{{mesh payload input must be an in payload record parameter}}
          out vertices MeshVertex vertices[3], out indices uint3 indices[1]) {
  SetMeshOutputCounts(3, 1);
  vertices[0].position = float4(0,0,0,1);
  indices[0] = uint3(0,0,0);
}
#endif