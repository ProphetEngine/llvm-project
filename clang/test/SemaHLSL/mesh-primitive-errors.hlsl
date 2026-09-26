// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-mesh -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify -DBAD_PRIMITIVE_TYPE %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-mesh -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify -DBAD_PRIMITIVE_COUNT %s

struct MeshVertex { float4 position : SV_Position; };
struct PrimitiveData { uint id : TEXCOORD1; };
[outputtopology("triangle")]
[numthreads(1,1,1)]
#ifdef BAD_PRIMITIVE_TYPE
void main(out vertices MeshVertex vertices[3], out indices uint3 triangles[1],
          out primitives uint primitive[1]) { // expected-error{{mesh primitives output must be a nonempty constant-size array of record values matching the indices array size}}
  SetMeshOutputCounts(3, 1);
  triangles[0] = uint3(0,1,2);
}
#else
void main(out vertices MeshVertex vertices[3], out indices uint3 triangles[1],
          out primitives PrimitiveData primitive[2]) { // expected-error{{mesh primitives output must be a nonempty constant-size array of record values matching the indices array size}}
  SetMeshOutputCounts(3, 1);
  triangles[0] = uint3(0,1,2);
}
#endif