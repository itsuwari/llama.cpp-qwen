#pragma once
#include <algorithm>
#include <cmath>
#include <cstdint>
#include <limits>
#if defined(__aarch64__)
#include <arm_neon.h>
#endif
// Return a unique finite argmax; ties and non-finite maxima use the original sampler.
static int32_t r95_unique_argmax(const float * z, int32_t n) {
    if (!z || n <= 0) return -1;
    float peak=-std::numeric_limits<float>::infinity();
    int32_t i=0;
#if defined(__aarch64__)
    float32x4_t a=vdupq_n_f32(peak),b=a;
    for (;i+8<=n;i+=8) { a=vmaxq_f32(a,vld1q_f32(z+i)); b=vmaxq_f32(b,vld1q_f32(z+i+4)); }
    peak=vmaxvq_f32(vmaxq_f32(a,b));
#endif
    for (;i<n;++i) { if (std::isnan(z[i])) return -1; peak=std::max(peak,z[i]); }
    if (!std::isfinite(peak)) return -1;
    int32_t first=-1,count=0; i=0;
#if defined(__aarch64__)
    const auto p=vdupq_n_f32(peak);
    for (;i+4<=n;i+=4) {
        const auto eq=vshrq_n_u32(vceqq_f32(vld1q_f32(z+i),p),31);
        const int c=int(vaddvq_u32(eq));
        if(c && first<0) for(int j=0;j<4;++j) if(z[i+j]==peak) {first=i+j;break;}
        count+=c;
        if(count>1) return -1;
    }
#endif
    for (;i<n;++i) if(z[i]==peak) { if(first<0) first=i; if(++count>1) return -1; }
    return count==1?first:-1;
}
