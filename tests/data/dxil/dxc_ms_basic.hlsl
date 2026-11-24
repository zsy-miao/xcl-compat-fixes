struct V { float4 pos : SV_Position; float4 color : COLOR0; };
struct P { uint id : SV_PrimitiveID; nointerpolation float2 extra : EXTRA; };
struct Pay { uint x; };
[outputtopology("triangle")]
[numthreads(32, 1, 1)]
void main(uint tid : SV_GroupThreadID, out vertices V verts[3], out indices uint3 tris[1], out primitives P prims[1]) {
  SetMeshOutputCounts(3, 1);
  if (tid < 3) { V v; v.pos = float4(tid, 0, 0, 1); v.color = float4(1,0,0,1); verts[tid] = v; }
  if (tid == 0) { tris[0] = uint3(0,1,2); P p; p.id = 7; p.extra = float2(1,2); prims[0] = p; }
}
