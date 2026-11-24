ByteAddressBuffer rawb : register(t0);
StructuredBuffer<uint2> sb : register(t1);
Buffer<float4> tb : register(t2);
Texture2D<float4> tex : register(t3, space1);
cbuffer CB : register(b0) { uint4 a; uint4 b; };
SamplerState smp : register(s0);
SamplerComparisonState cmp : register(s1);
RWByteAddressBuffer res : register(u1);
[numthreads(32,1,1)]
void main(uint id : SV_GroupIndex) {
  uint v = rawb.Load(id*4) + sb[id].y + asuint(tb[id].x) + b.y;
  float f = tex.SampleLevel(smp, float2(0.5,0.5), 0).x + tex.SampleCmpLevelZero(cmp, float2(0.5,0.5), 0.5);
  res.Store(id*4, v + asuint(f));
}
