#pragma once
#include <cmath>
#include <cstddef>
#include <cstdint>
#include <limits>

// Return an exact, unique finite maximum. Negative infinity may mask tokens.
// Ties, NaNs, positive infinity, and empty/all-masked rows require the caller's
// original sampler. This helper never changes logits or token ordering.
inline int32_t common_sampler_argmax_unique(const float * logits, size_t size) noexcept {
    if (!logits || size == 0 || size > size_t(std::numeric_limits<int32_t>::max())) return -1;
    float best = -std::numeric_limits<float>::infinity();
    int32_t selected = -1;
    bool tied = false, has_nan = false;
    for (size_t i = 0; i < size; ++i) {
        const float value = logits[i];
        has_nan |= std::isnan(value);
        if (value > best) {
            best = value;
            selected = int32_t(i);
            tied = false;
        } else if (value == best) {
            tied = true;
        }
    }
    return selected >= 0 && !tied && !has_nan && std::isfinite(best) ? selected : -1;
}
