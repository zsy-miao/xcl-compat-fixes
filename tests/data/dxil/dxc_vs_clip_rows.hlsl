struct VSOut { float4 pos : SV_Position; float a : A; float clip : SV_ClipDistance0; float b : B; float cull : SV_CullDistance0; float c : C; float2 d : D; };
VSOut main(float4 p : POSITION) { VSOut o; o.pos = p; o.a = p.x; o.clip = p.y; o.b = p.z; o.cull = p.w; o.c = 1; o.d = 2; return o; }
