RWByteAddressBuffer mem : register(u2);
float4 main(uint vid : SV_VertexID) : SV_Position { return asfloat(mem.Load4(vid * 16)); }
