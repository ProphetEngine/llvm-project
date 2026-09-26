// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-hull -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main -DHULL \
// RUN:   -fsyntax-only -verify %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-domain -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify %s

// expected-no-diagnostics

struct ControlPoint { float4 position : SV_Position; };
struct PatchData {
  float edges[3] : SV_TessFactor;
  float inside : SV_InsideTessFactor;
};

#ifdef HULL
PatchData constants(InputPatch<ControlPoint, 3> patch) {
  PatchData result;
  result.edges[0] = result.edges[1] = result.edges[2] = 1.0;
  result.inside = 1.0;
  return result;
}
[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("constants")]
ControlPoint main(InputPatch<ControlPoint, 3> patch,
                  uint id : SV_OutputControlPointID) {
  return patch[id];
}
#else
[domain("tri")]
float4 main(PatchData factors, float3 bary : SV_DomainLocation,
            const OutputPatch<ControlPoint, 3> patch) : SV_Position {
  return patch[0].position * bary.x + patch[1].position * bary.y +
         patch[2].position * bary.z;
}
#endif
