struct V { float4 pos : SV_Position; };
struct P { nointerpolation uint k : PRIMK; bool cull : SV_CullPrimitive; uint rt : SV_RenderTargetArrayIndex; };
struct Pay { uint a; float b[3]; };
groupshared uint gs[16];
[outputtopology("line")]
[numthreads(16, 2, 1)]
void main(uint tid : SV_GroupIndex, in payload Pay pay, out vertices V verts[4], out indices uint2 lines[2], out primitives P prims[2]) {
  gs[tid % 16] = pay.a;
  GroupMemoryBarrierWithGroupSync();
  SetMeshOutputCounts(4, 2);
  if (tid < 4) { verts[tid].pos = float4(gs[(tid + 1) % 16], pay.b[tid % 3], 0, 1); }
  if (tid < 2) { lines[tid] = uint2(tid, tid + 1); prims[tid].k = tid; prims[tid].cull = false; prims[tid].rt = 1; }
}
