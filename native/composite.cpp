#include <cstddef>
#include <cmath>
#ifdef _WIN32
#define EXPORT __declspec(dllexport)
#else
#define EXPORT __attribute__((visibility("default")))
#endif
// Caller owns all buffers. Three channels per pixel, contiguous float32.
// Clamp finite values while preserving NaNs like NumPy clip.
static float clip(float x) { return x < 0.f ? 0.f : (x > 1.f ? 1.f : x); }
extern "C" EXPORT void portrait_composite_v2(const float* fg, const float* alpha,
                                         const float* bg, float* out, std::size_t pixels, float strength) {
    for (std::size_t p = 0; p < pixels; ++p) {
        const float a = 1.f - strength * (1.f - clip(alpha[p]));
        for (std::size_t c = 0; c < 3; ++c) {
            const auto i = p * 3 + c;
            out[i] = clip(fg[i] * a + bg[i] * (1.f - a));
        }
    }
}
