struct VSOut { float4 pos : SV_Position; float2 a : A; float3 clip : SV_ClipDistance0; float c : C; float2 cull : SV_CullDistance0; float clip1 : SV_ClipDistance1; nointerpolation uint vp : SV_ViewportArrayIndex; };
VSOut main(float4 p : POSITION) { VSOut o; o.pos = p; o.a = p.xy; o.clip = p.xyz; o.c = p.w; o.cull = p.zw; o.clip1 = p.x; o.vp = 1; return o; }
