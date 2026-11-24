float4 main(float4 pos : SV_Position, float4 color : COLOR0, float2 uv : TEXCOORD1) : SV_Target { return color + uv.xyxy; }
