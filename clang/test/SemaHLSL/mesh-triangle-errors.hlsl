// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-mesh -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify -DMISSING_TOPOLOGY %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-mesh -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify -DBAD_INDICES %s

struct MeshVertex { float4 position : SV_Position; float4 color : COLOR0; };
#ifdef MISSING_TOPOLOGY
[numthreads(1,1,1)]
void main(out vertices MeshVertex vertices[3], // expected-error{{mesh shader entry requires an outputtopology attribute}}
          out indices uint3 triangles[1]) {
  SetMeshOutputCounts(3, 1);
  triangles[0] = uint3(0,1,2);
}
#else
[outputtopology("triangle")]
[numthreads(1,1,1)]
void main(out vertices MeshVertex vertices[3],
          out indices uint2 triangles[1]) { // expected-error{{mesh indices output must be a nonempty constant-size array of uint3 values}}
  SetMeshOutputCounts(3, 1);
  triangles[0] = uint2(0,1);
}
#endif