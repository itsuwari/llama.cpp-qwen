#include "llama.h"
#include "greedy-argmax.h"
#include <chrono>
#include <cmath>
#include <cstdio>
#include <memory>
#include <random>
#include <stdexcept>
#include <vector>

using sampler_ptr = std::unique_ptr<llama_sampler, decltype(&llama_sampler_free)>;

static sampler_ptr make_chain() {
    auto p = llama_sampler_chain_default_params();
    p.no_perf = true;
    sampler_ptr s(llama_sampler_chain_init(p), llama_sampler_free);
    llama_sampler_chain_add(s.get(), llama_sampler_init_top_k(40));
    llama_sampler_chain_add(s.get(), llama_sampler_init_top_p(0.95f, 1));
    llama_sampler_chain_add(s.get(), llama_sampler_init_min_p(0.05f, 1));
    llama_sampler_chain_add(s.get(), llama_sampler_init_temp(0.0f));
    llama_sampler_chain_add(s.get(), llama_sampler_init_dist(1234));
    return s;
}

int main() {
    try {
        constexpr size_t nv = 248320;
        constexpr int reps = 512;
        std::mt19937 rng(90123);
        std::normal_distribution<float> normal(0.0f, 3.0f);
        std::vector<float> logits(nv);
        std::vector<llama_token_data> candidates(nv);
        size_t fallbacks = 0;
        const auto select = [&](llama_sampler * s, bool fast) {
            if (fast) {
                const int32_t id = common_sampler_argmax_unique(logits.data(), logits.size());
                if (id >= 0) return llama_token(id);
                ++fallbacks;
            }
            for (size_t i = 0; i < nv; ++i) candidates[i] = {llama_token(i), logits[i], 0.0f};
            llama_token_data_array a{candidates.data(), nv, -1, false};
            llama_sampler_apply(s, &a);
            if (a.selected < 0 || size_t(a.selected) >= a.size) throw std::runtime_error("invalid selection");
            return a.data[a.selected].id;
        };
        auto reference = make_chain(), optimized = make_chain();
        size_t checks = 0;
        for (int trial = 0; trial < 100; ++trial) {
            for (auto & v : logits) v = normal(rng);
            if (trial % 4 == 0) { logits[7] = 30.0f; logits[9001] = 30.0f; }
            if (trial % 4 == 1) for (size_t i = 0; i < nv; i += 3) logits[i] = -INFINITY;
            if (trial % 4 == 2) { logits[0] = -1000.0f; logits[nv-1] = 1000.0f; }
            if (select(reference.get(), false) != select(optimized.get(), true)) throw std::runtime_error("guarded selection mismatch");
            ++checks;
        }
        const float exceptional[][3] = {{NAN, 1, 2}, {INFINITY, 1, 2}, {-INFINITY, -INFINITY, -INFINITY}, {2, 1, 2}};
        for (const auto & row : exceptional) {
            if (common_sampler_argmax_unique(row, 3) != -1) throw std::runtime_error("missing exceptional-row fallback");
            ++checks;
        }
        if (common_sampler_argmax_unique(nullptr, 0) != -1) throw std::runtime_error("empty-row guard failed");
        ++checks;
        const size_t checked_fallbacks = fallbacks;
        for (int order = 0; order < 4; ++order) {
            const bool fast = order % 2 != 0;
            auto * s = fast ? optimized.get() : reference.get();
            uint64_t checksum = 0;
            const auto begin = std::chrono::steady_clock::now();
            for (int i = 0; i < reps; ++i) checksum += uint64_t(select(s, fast));
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-begin).count();
            std::printf("SAMPLING_COST mode=%s vocab=%zu reps=%d us_per_token=%.3f checksum=%llu\n",
                fast ? "guarded_unique_max" : "reference", nv, reps, us/reps, (unsigned long long)checksum);
        }
        std::printf("SAMPLING_CHECK_PASS cases=%zu tie_fallbacks=%zu\n", checks, checked_fallbacks);
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "SAMPLING_ERROR %s\n", e.what());
        return 1;
    }
}
