struct O { float4 a : SV_Target0; float4 b : SV_Target1; float d : SV_Depth; };
O main(float4 pos : SV_Position, nointerpolation uint m : MODE, centroid float2 c : TEXCOORD0, noperspective float t : TEXCOORD1, sample float3 s : TEXCOORD2, bool ff : SV_IsFrontFace, uint pid : SV_PrimitiveID, uint cov : SV_Coverage, linear centroid noperspective float q : Q) {
  O o; if (m == 3) discard; o.a = float4(c, t, s.x); o.b = float4(s.yz, ff ? 1 : 0, pid + cov + q); o.d = pos.z; return o; }
