#include "llama.h"
#include "llama-ext.h"
#include "ggml-backend.h"
#include "nlohmann/json.hpp"
#include <chrono>
#include <cstdio>
#include <fstream>
#include <memory>
#include <stdexcept>
#include <vector>

using json = nlohmann::ordered_json;

static json load_json(const char * path) {
    std::ifstream stream(path);
    if (!stream) throw std::runtime_error("fixture open failed");
    return json::parse(stream);
}

int main(int argc, char ** argv) {
    try {
        if (argc != 4) throw std::runtime_error("usage: target-replay model.gguf prompt.json result.json");
        auto tokens = load_json(argv[2]).get<std::vector<llama_token>>();
        if (tokens.size() != 4096) throw std::runtime_error("prompt must have exactly 4096 tokens");
        const auto generated = load_json(argv[3]).at("runs").at(0).at("tokens").get<std::vector<llama_token>>();
        if (generated.size() < 256) throw std::runtime_error("fixture needs at least 256 output tokens");
        tokens.insert(tokens.end(), generated.begin(), generated.end());
        ggml_backend_load_all();
        llama_backend_init();
        auto mp = llama_model_default_params();
        mp.n_gpu_layers = 999;
        std::unique_ptr<llama_model, decltype(&llama_model_free)> model(llama_model_load_from_file(argv[1], mp), llama_model_free);
        if (!model) throw std::runtime_error("model load failed");
        const auto * vocab = llama_model_get_vocab(model.get());
        const int nv = llama_vocab_n_tokens(vocab);
        for (auto t : tokens) if (t < 0 || t >= nv) throw std::runtime_error("token outside vocabulary");
        auto cp = llama_context_default_params();
        cp.n_ctx = 8192; cp.n_batch = 512; cp.n_ubatch = 512;
        cp.n_seq_max = 1; cp.n_rs_seq = 3; cp.n_threads = 24; cp.n_threads_batch = 24;
        cp.n_outputs_max = 4; cp.n_outputs_max_per_seq = 4;
        cp.type_k = GGML_TYPE_Q8_0; cp.type_v = GGML_TYPE_Q8_0;
        cp.flash_attn_type = LLAMA_FLASH_ATTN_TYPE_ENABLED;
        std::unique_ptr<llama_context, decltype(&llama_free)> ctx(llama_init_from_model(model.get(), cp), llama_free);
        if (!ctx) throw std::runtime_error("context load failed");
        llama_set_embeddings_nextn(ctx.get(), true, false);
        auto mem = llama_get_memory(ctx.get());
        std::vector<llama_pos> positions(512);
        std::vector<int8_t> outputs(512);
        const auto decode = [&](int pos, int n, bool logits) {
            for (int i = 0; i < n; ++i) { positions[i] = pos+i; outputs[i] = logits ? 1 : 0; }
            llama_batch batch{};
            batch.n_tokens = n; batch.token = tokens.data()+pos; batch.pos = positions.data(); batch.logits = outputs.data();
            if (llama_decode(ctx.get(), batch) != 0) throw std::runtime_error("decode failed");
            llama_synchronize(ctx.get());
        };
        for (int pos = 0; pos < 4096; pos += 512) decode(pos, 512, false);
        constexpr int width = 4, keep = 3, rounds = 64, warmup = 4;
        json samples = json::array();
        int pos = 4096;
        for (int round = 0; round < rounds+warmup; ++round) {
            const auto begin = std::chrono::steady_clock::now();
            decode(pos, width, true);
            if (!llama_memory_seq_rm(mem, 0, pos+keep, -1)) throw std::runtime_error("suffix rollback failed");
            const double us = std::chrono::duration<double, std::micro>(std::chrono::steady_clock::now()-begin).count();
            if (round >= warmup) samples.push_back(us);
            pos += keep;
        }
        double total = 0;
        for (const auto & t : samples) total += t.get<double>();
        json result{{"scope", "target-only fixed-token replay; excludes drafting and sampling"},
                    {"prompt_tokens",4096},{"vocab",nv},{"width",width},{"keep",keep},{"n_rs_seq",3},
                    {"rounds",rounds},{"mean_round_us",total/rounds},{"round_us",samples}};
        std::printf("TARGET_REPLAY %s\n", result.dump().c_str());
        return 0;
    } catch (const std::exception & e) {
        std::fprintf(stderr, "TARGET_REPLAY_ERROR %s\n", e.what());
        return 1;
    }
}
