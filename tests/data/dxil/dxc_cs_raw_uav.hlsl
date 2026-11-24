RWByteAddressBuffer io : register(u0);
RWByteAddressBuffer res : register(u1);
[WaveSize(64)]
[numthreads(64,1,1)]
void main(uint3 id : SV_GroupThreadID) {
  uint a = io.Load(id.x*4);
  float f = asfloat(a) * 2.0f + 1.0f;
  if (a > 5) f = -f;
  res.Store(id.x*4, asuint(f) + WaveReadLaneAt(a, 3));
}
