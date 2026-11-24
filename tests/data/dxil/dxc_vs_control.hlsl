struct VSOut { float4 pos : SV_Position; float a : A; float b : B; };
VSOut main(float4 p : POSITION, float c : C, uint iid : SV_InstanceID) {
  VSOut o; o.pos = p; o.a = 0; o.b = 1;
  [branch] if (c > 0.5) { o.a = p.x * 2; } 
  [loop] for (uint i = 0; i < iid; i++) { o.b += p.y; }
  return o; }
