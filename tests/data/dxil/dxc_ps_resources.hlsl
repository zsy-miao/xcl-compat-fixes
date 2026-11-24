cbuffer RC : register(b2, space1) { uint k; float scale; };
Texture2D<float4> tex : register(t0);
SamplerState smp : register(s3);
float4 main(float4 p : SV_Position) : SV_Target { return tex.Sample(smp, p.xy * scale) + (float)k; }
