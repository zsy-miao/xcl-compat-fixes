struct VSOut { float4 pos : SV_Position; float4 color : COLOR0; float2 uv : TEXCOORD1; };
VSOut main(float3 p : POSITION, float4 c : COLOR, uint vid : SV_VertexID) {
  VSOut o; o.pos = float4(p, 1); o.color = c; o.uv = float2(vid, 0); return o; }
