// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-hull -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify -DBAD_HULL_FACTOR %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-hull -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify -DBAD_HELPER_PATCH %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-domain -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify -DBAD_DOMAIN_FACTOR %s
// RUN: %clang_cc1 -triple spirv-unknown-vulkan1.3-domain -x hlsl \
// RUN:   -std=hlsl202x -finclude-default-header -hlsl-entry main \
// RUN:   -fsyntax-only -verify -DBAD_DOMAIN_LOCATION %s

struct ControlPoint { float4 position : SV_Position; };

#if defined(BAD_HULL_FACTOR)
struct PatchData {
  float edges[2] : SV_TessFactor;
  float inside : SV_InsideTessFactor;
};
PatchData constants(InputPatch<ControlPoint, 3> patch) {
  PatchData result;
  result.edges[0] = result.edges[1] = result.inside = 1.0f;
  return result;
}
[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("constants")]
ControlPoint main(InputPatch<ControlPoint, 3> patch, // expected-error{{invalid SV_TessFactor or SV_InsideTessFactor shape for tri domain}}
                  uint id : SV_OutputControlPointID) {
  return patch[id];
}
#elif defined(BAD_HELPER_PATCH)
struct PatchData {
  float edges[3] : SV_TessFactor;
  float inside : SV_InsideTessFactor;
};
PatchData constants(InputPatch<ControlPoint, 2> patch) {
  PatchData result;
  result.edges[0] = result.edges[1] = result.edges[2] =
      result.inside = 1.0f;
  return result;
}
[domain("tri")]
[partitioning("integer")]
[outputtopology("triangle_cw")]
[outputcontrolpoints(3)]
[patchconstantfunc("constants")]
ControlPoint main(InputPatch<ControlPoint, 3> patch, // expected-error{{patchconstantfunc must name a defined function returning patch constants}}
                  uint id : SV_OutputControlPointID) {
  return patch[id];
}
#elif defined(BAD_DOMAIN_FACTOR)
struct PatchData {
  float edges[4] : SV_TessFactor;
  float inside : SV_InsideTessFactor;
};
[domain("tri")]
float4 main(PatchData factors, float3 bary : SV_DomainLocation, // expected-error{{invalid SV_TessFactor or SV_InsideTessFactor shape for tri domain}}
            const OutputPatch<ControlPoint, 3> patch) : SV_Position {
  return patch[0].position;
}
#elif defined(BAD_DOMAIN_LOCATION)
struct PatchData {
  float edges[3] : SV_TessFactor;
  float inside : SV_InsideTessFactor;
};
[domain("tri")]
float4 main(PatchData factors, float2 bary : SV_DomainLocation, // expected-error{{invalid hull or domain shader entry signature}}
            const OutputPatch<ControlPoint, 3> patch) : SV_Position {
  return patch[0].position;
}
#endif