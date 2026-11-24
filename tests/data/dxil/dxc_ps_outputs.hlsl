struct O { float4 c : SV_Target2; uint cov : SV_Coverage; float d : SV_DepthGreaterEqual; uint s : SV_StencilRef; };
O main(noperspective centroid float4 p : SV_Position, nointerpolation uint k : PRIMK, uint rt : SV_RenderTargetArrayIndex) { O o; o.c = float4(k, rt, 0, 1); if (p.x > 4) o.c.z = 2; o.cov = 1; o.d = 0.5; o.s = k; return o; }
