#include <metal_stdlib>
using namespace metal;
kernel void portraitBlend(texture2d<float, access::read> frame [[texture(0)]],
                          texture2d<float, access::sample> mask [[texture(1)]],
                          texture2d<float, access::write> output [[texture(2)]],
                          uint2 p [[thread_position_in_grid]]) {
    if (p.x >= output.get_width() || p.y >= output.get_height()) return;
    constexpr sampler s(coord::normalized, address::clamp_to_edge, filter::linear);
    float2 uv=(float2(p)+0.5)/float2(output.get_width(),output.get_height());
    float alpha=clamp(mask.sample(s,uv).r,0.0,1.0);
    float3 background=float3(0.12,0.16,0.22);
    output.write(float4(mix(background,frame.read(p).rgb,alpha),1.0),p);
}
