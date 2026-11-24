struct VSOut { float2 a : A; float2 b : B; float c : C; nointerpolation uint d : D; float4 pos : SV_Position; float e : E; nointerpolation uint rt : SV_RenderTargetArrayIndex; nointerpolation uint2 f : F; float3 g : G; float2x2 m : M; uint i : I; };
VSOut main(uint vid : SV_VertexID, uint iid : SV_InstanceID) {
  VSOut o; o.a = vid; o.b = 1; o.c = 2; o.d = iid; o.pos = 0; o.e = 3; o.rt = vid; o.f = iid; o.g = 4; o.m = float2x2(1,2,3,vid); o.i = 5; return o; }
