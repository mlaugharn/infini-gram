// g++ -std=c++20 -O3 -pthread test/cpp_engine_perf_bench.cpp -o test/cpp_engine_perf_bench
// ./test/cpp_engine_perf_bench --index-dir /path/to/index/v4_pileval_llama

#include "../infini_gram/cpp_engine.h"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <filesystem>
#include <fstream>
#include <iomanip>
#include <iostream>
#include <limits>
#include <numeric>
#include <sstream>
#include <unordered_set>

using Clock = std::chrono::steady_clock;

struct BenchConfig {
    std::string index_dir;
    size_t sample_tokens = 8'000'000; // if 0, sample uniformly from full tokenized.0
    size_t queries_per_len = 1500;
    size_t warmup_n = 200;
    size_t batch_size = 32;
    size_t runs = 3;
    uint32_t seed = 19260817;
    std::vector<size_t> lengths = {2, 4, 8, 16, 32};
};

struct DistStats {
    size_t n = 0;
    double min = 0;
    double p50 = 0;
    double p95 = 0;
    double p99 = 0;
    double max = 0;
    double mean = 0;
};

struct OpResult {
    DistStats latency_us;
    double total_ms = 0;
    double qps = 0;
    uint64_t checksum = 0;
    size_t num_queries = 0;
    size_t num_hits = 0; // count/prob-specific: #queries with non-zero support
};

static void print_usage(const char* argv0) {
    std::cerr
        << "Usage: " << argv0 << " --index-dir <dir> [options]\n"
        << "Options:\n"
        << "  --sample-tokens <n>    default: 8000000 (0 => full-corpus random sampling)\n"
        << "  --queries-per-len <n>  default: 1500\n"
        << "  --lengths <csv>        default: 2,4,8,16,32\n"
        << "  --warmup-n <n>         default: 200\n"
        << "  --batch-size <n>       default: 32\n"
        << "  --runs <n>             default: 3\n"
        << "  --seed <n>             default: 19260817\n";
}

static bool parse_size_arg(const std::string& s, size_t* out) {
    try {
        *out = static_cast<size_t>(std::stoull(s));
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_u32_arg(const std::string& s, uint32_t* out) {
    try {
        unsigned long long v = std::stoull(s);
        if (v > std::numeric_limits<uint32_t>::max()) return false;
        *out = static_cast<uint32_t>(v);
        return true;
    } catch (...) {
        return false;
    }
}

static bool parse_lengths_csv(const std::string& s, std::vector<size_t>* out) {
    std::vector<size_t> vals;
    std::stringstream ss(s);
    std::string item;
    while (std::getline(ss, item, ',')) {
        if (item.empty()) continue;
        size_t v = 0;
        if (!parse_size_arg(item, &v) || v < 2) {
            return false;
        }
        vals.push_back(v);
    }
    if (vals.empty()) return false;
    std::sort(vals.begin(), vals.end());
    vals.erase(std::unique(vals.begin(), vals.end()), vals.end());
    *out = std::move(vals);
    return true;
}

static bool parse_args(int argc, char** argv, BenchConfig* cfg) {
    for (int i = 1; i < argc; i++) {
        const std::string arg = argv[i];

        auto need_value = [&](const char* name) -> const char* {
            if (i + 1 >= argc) {
                std::cerr << "Missing value for " << name << "\n";
                return nullptr;
            }
            return argv[++i];
        };

        if (arg == "--index-dir") {
            const char* v = need_value("--index-dir");
            if (!v) return false;
            cfg->index_dir = v;
        } else if (arg == "--sample-tokens") {
            const char* v = need_value("--sample-tokens");
            if (!v || !parse_size_arg(v, &cfg->sample_tokens)) return false;
        } else if (arg == "--queries-per-len") {
            const char* v = need_value("--queries-per-len");
            if (!v || !parse_size_arg(v, &cfg->queries_per_len)) return false;
        } else if (arg == "--lengths") {
            const char* v = need_value("--lengths");
            if (!v || !parse_lengths_csv(v, &cfg->lengths)) return false;
        } else if (arg == "--warmup-n") {
            const char* v = need_value("--warmup-n");
            if (!v || !parse_size_arg(v, &cfg->warmup_n)) return false;
        } else if (arg == "--batch-size") {
            const char* v = need_value("--batch-size");
            if (!v || !parse_size_arg(v, &cfg->batch_size)) return false;
        } else if (arg == "--runs") {
            const char* v = need_value("--runs");
            if (!v || !parse_size_arg(v, &cfg->runs)) return false;
        } else if (arg == "--seed") {
            const char* v = need_value("--seed");
            if (!v || !parse_u32_arg(v, &cfg->seed)) return false;
        } else if (arg == "--help" || arg == "-h") {
            print_usage(argv[0]);
            std::exit(0);
        } else {
            std::cerr << "Unknown argument: " << arg << "\n";
            return false;
        }
    }

    if (cfg->index_dir.empty()) {
        std::cerr << "--index-dir is required\n";
        return false;
    }
    if (cfg->queries_per_len == 0 || cfg->batch_size == 0 || cfg->runs == 0) {
        std::cerr << "queries-per-len, batch-size, and runs must be > 0\n";
        return false;
    }
    return true;
}

static size_t get_total_tokens_in_file(const std::string& tokenized_path) {
    const uintmax_t bytes = std::filesystem::file_size(tokenized_path);
    if (bytes % sizeof(U16) != 0) {
        throw std::runtime_error("tokenized file size is not divisible by token width");
    }
    return static_cast<size_t>(bytes / sizeof(U16));
}

static std::vector<U16> load_tokens_prefix(const std::string& tokenized_path, size_t max_tokens) {
    std::ifstream f(tokenized_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("failed to open tokenized file: " + tokenized_path);
    }
    std::vector<U16> tokens(max_tokens);
    f.read(reinterpret_cast<char*>(tokens.data()), static_cast<std::streamsize>(max_tokens * sizeof(U16)));
    size_t got_tokens = static_cast<size_t>(f.gcount()) / sizeof(U16);
    tokens.resize(got_tokens);
    return tokens;
}

static std::vector<std::vector<U16>> build_fixed_len_queries_from_prefix(
    const std::vector<U16>& tokens,
    size_t n_queries,
    size_t len,
    U16 doc_sep_id,
    uint32_t seed) {

    std::vector<std::vector<U16>> queries;
    queries.reserve(n_queries);
    if (tokens.size() <= len + 1) return queries;

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> start_dist(0, tokens.size() - len - 1);

    size_t attempts = 0;
    const size_t max_attempts = n_queries * 100;
    while (queries.size() < n_queries && attempts < max_attempts) {
        attempts++;
        const size_t start = start_dist(rng);

        bool has_doc_sep = false;
        for (size_t i = 0; i < len; i++) {
            if (tokens[start + i] == doc_sep_id) {
                has_doc_sep = true;
                break;
            }
        }
        if (has_doc_sep) continue;

        queries.emplace_back(tokens.begin() + static_cast<std::ptrdiff_t>(start),
                             tokens.begin() + static_cast<std::ptrdiff_t>(start + len));
    }

    return queries;
}

static std::vector<std::vector<U16>> build_fixed_len_queries_from_file(
    const std::string& tokenized_path,
    size_t total_tokens,
    size_t n_queries,
    size_t len,
    U16 doc_sep_id,
    uint32_t seed) {

    std::vector<std::vector<U16>> queries;
    queries.reserve(n_queries);
    if (total_tokens <= len + 1) return queries;

    std::ifstream f(tokenized_path, std::ios::binary);
    if (!f) {
        throw std::runtime_error("failed to open tokenized file: " + tokenized_path);
    }

    std::mt19937 rng(seed);
    std::uniform_int_distribution<size_t> start_dist(0, total_tokens - len - 1);
    std::vector<U16> buf(len);

    size_t attempts = 0;
    const size_t max_attempts = n_queries * 200;
    while (queries.size() < n_queries && attempts < max_attempts) {
        attempts++;
        const size_t start = start_dist(rng);
        const std::streamoff byte_off = static_cast<std::streamoff>(start * sizeof(U16));
        f.seekg(byte_off, std::ios::beg);
        if (!f.good()) continue;
        f.read(reinterpret_cast<char*>(buf.data()), static_cast<std::streamsize>(len * sizeof(U16)));
        if (f.gcount() != static_cast<std::streamsize>(len * sizeof(U16))) {
            f.clear();
            continue;
        }

        bool has_doc_sep = false;
        for (U16 tok : buf) {
            if (tok == doc_sep_id) {
                has_doc_sep = true;
                break;
            }
        }
        if (has_doc_sep) continue;

        queries.push_back(buf);
    }

    return queries;
}

static DistStats summarize(std::vector<double> xs) {
    DistStats s;
    if (xs.empty()) return s;
    std::sort(xs.begin(), xs.end());
    s.n = xs.size();
    s.min = xs.front();
    s.max = xs.back();
    s.mean = std::accumulate(xs.begin(), xs.end(), 0.0) / static_cast<double>(xs.size());

    auto percentile = [&](double p) {
        if (xs.size() == 1) return xs[0];
        double pos = p * static_cast<double>(xs.size() - 1);
        size_t i = static_cast<size_t>(std::floor(pos));
        size_t j = static_cast<size_t>(std::ceil(pos));
        double frac = pos - static_cast<double>(i);
        return xs[i] * (1.0 - frac) + xs[j] * frac;
    };

    s.p50 = percentile(0.50);
    s.p95 = percentile(0.95);
    s.p99 = percentile(0.99);
    return s;
}

static std::string join_lengths(const std::vector<size_t>& lengths) {
    std::ostringstream oss;
    for (size_t i = 0; i < lengths.size(); i++) {
        if (i) oss << ",";
        oss << lengths[i];
    }
    return oss.str();
}

static uint64_t hash_query(const std::vector<U16>& q) {
    // FNV-1a over bytes
    uint64_t h = 1469598103934665603ull;
    const U8* b = reinterpret_cast<const U8*>(q.data());
    size_t n = q.size() * sizeof(U16);
    for (size_t i = 0; i < n; i++) {
        h ^= static_cast<uint64_t>(b[i]);
        h *= 1099511628211ull;
    }
    return h;
}

static void print_op_result(const std::string& scale, const std::string& op, const OpResult& r) {
    std::cout << std::fixed << std::setprecision(4)
              << "RESULT scale=" << scale
              << " op=" << op
              << " n=" << r.num_queries
              << " total_ms=" << r.total_ms
              << " qps=" << r.qps
              << " us_min=" << r.latency_us.min
              << " us_p50=" << r.latency_us.p50
              << " us_p95=" << r.latency_us.p95
              << " us_p99=" << r.latency_us.p99
              << " us_max=" << r.latency_us.max
              << " us_mean=" << r.latency_us.mean
              << " hits=" << r.num_hits
              << " checksum=" << r.checksum
              << "\n";
}

static OpResult bench_count(const Engine<U16>& engine, const std::vector<std::vector<U16>>& queries) {
    std::vector<double> lat_us;
    lat_us.reserve(queries.size());
    uint64_t checksum = 0;
    size_t hits = 0;

    auto t_all0 = Clock::now();
    for (const auto& q : queries) {
        auto t0 = Clock::now();
        auto r = engine.count(q);
        auto t1 = Clock::now();
        lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        checksum += r.count;
        if (r.count > 0) hits++;
    }
    auto t_all1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();

    OpResult out;
    out.latency_us = summarize(std::move(lat_us));
    out.total_ms = total_ms;
    out.qps = static_cast<double>(queries.size()) / (total_ms / 1000.0);
    out.checksum = checksum;
    out.num_queries = queries.size();
    out.num_hits = hits;
    return out;
}

static OpResult bench_find(const Engine<U16>& engine, const std::vector<std::vector<U16>>& queries) {
    std::vector<double> lat_us;
    lat_us.reserve(queries.size());
    uint64_t checksum = 0;
    size_t hits = 0;

    auto t_all0 = Clock::now();
    for (const auto& q : queries) {
        auto t0 = Clock::now();
        auto r = engine.find(q);
        auto t1 = Clock::now();
        lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        checksum += r.cnt;
        if (!r.segment_by_shard.empty()) checksum += r.segment_by_shard[0].first;
        if (r.cnt > 0) hits++;
    }
    auto t_all1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();

    OpResult out;
    out.latency_us = summarize(std::move(lat_us));
    out.total_ms = total_ms;
    out.qps = static_cast<double>(queries.size()) / (total_ms / 1000.0);
    out.checksum = checksum;
    out.num_queries = queries.size();
    out.num_hits = hits;
    return out;
}

static OpResult bench_prob(const Engine<U16>& engine, const std::vector<std::vector<U16>>& queries) {
    std::vector<double> lat_us;
    lat_us.reserve(queries.size());
    uint64_t checksum = 0;
    size_t hits = 0;

    auto t_all0 = Clock::now();
    for (const auto& q : queries) {
        std::vector<U16> prompt(q.begin(), q.end() - 1);
        U16 cont_id = q.back();

        auto t0 = Clock::now();
        auto r = engine.prob(prompt, cont_id);
        auto t1 = Clock::now();

        lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        checksum += r.prompt_cnt + r.cont_cnt;
        if (r.prompt_cnt > 0) hits++;
    }
    auto t_all1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();

    OpResult out;
    out.latency_us = summarize(std::move(lat_us));
    out.total_ms = total_ms;
    out.qps = static_cast<double>(queries.size()) / (total_ms / 1000.0);
    out.checksum = checksum;
    out.num_queries = queries.size();
    out.num_hits = hits;
    return out;
}

static OpResult bench_prob_batched(const Engine<U16>& engine, const std::vector<std::vector<U16>>& queries, size_t batch_size) {
    std::vector<double> lat_us_per_query;
    lat_us_per_query.reserve(queries.size());
    uint64_t checksum = 0;
    size_t hits = 0;

    auto t_all0 = Clock::now();
    for (size_t i = 0; i < queries.size(); i += batch_size) {
        size_t j = std::min(i + batch_size, queries.size());
        std::vector<std::vector<U16>> prompt_batch;
        std::vector<U16> cont_ids;
        prompt_batch.reserve(j - i);
        cont_ids.reserve(j - i);
        for (size_t k = i; k < j; k++) {
            prompt_batch.emplace_back(queries[k].begin(), queries[k].end() - 1);
            cont_ids.push_back(queries[k].back());
        }

        auto t0 = Clock::now();
        auto rs = engine.prob_batched(prompt_batch, cont_ids);
        auto t1 = Clock::now();
        double per_q_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(j - i);
        for (size_t k = i; k < j; k++) lat_us_per_query.push_back(per_q_us);

        for (const auto& r : rs) {
            checksum += r.prompt_cnt + r.cont_cnt;
            if (r.prompt_cnt > 0) hits++;
        }
    }
    auto t_all1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();

    OpResult out;
    out.latency_us = summarize(std::move(lat_us_per_query));
    out.total_ms = total_ms;
    out.qps = static_cast<double>(queries.size()) / (total_ms / 1000.0);
    out.checksum = checksum;
    out.num_queries = queries.size();
    out.num_hits = hits;
    return out;
}

static OpResult bench_prob_sequence(const Engine<U16>& engine, const std::vector<std::vector<U16>>& queries) {
    std::vector<double> lat_us;
    lat_us.reserve(queries.size());
    uint64_t checksum = 0;
    size_t hits = 0;

    auto t_all0 = Clock::now();
    for (const auto& q : queries) {
        auto t0 = Clock::now();
        auto rs = engine.prob_sequence(q);
        auto t1 = Clock::now();

        lat_us.push_back(std::chrono::duration<double, std::micro>(t1 - t0).count());
        for (const auto& r : rs) {
            checksum += r.prompt_cnt + r.cont_cnt;
            if (r.prompt_cnt > 0) hits++;
        }
    }
    auto t_all1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();

    OpResult out;
    out.latency_us = summarize(std::move(lat_us));
    out.total_ms = total_ms;
    out.qps = static_cast<double>(queries.size()) / (total_ms / 1000.0);
    out.checksum = checksum;
    out.num_queries = queries.size();
    out.num_hits = hits;
    return out;
}

static OpResult bench_prob_batched_sequence(const Engine<U16>& engine, const std::vector<std::vector<U16>>& queries, size_t batch_size) {
    std::vector<double> lat_us_per_query;
    lat_us_per_query.reserve(queries.size());
    uint64_t checksum = 0;
    size_t hits = 0;

    auto t_all0 = Clock::now();
    for (size_t i = 0; i < queries.size(); i += batch_size) {
        size_t j = std::min(i + batch_size, queries.size());
        std::vector<std::vector<U16>> batch;
        batch.reserve(j - i);
        for (size_t k = i; k < j; k++) batch.push_back(queries[k]);

        auto t0 = Clock::now();
        auto rss = engine.prob_batched_sequence(batch);
        auto t1 = Clock::now();
        double per_q_us = std::chrono::duration<double, std::micro>(t1 - t0).count() / static_cast<double>(j - i);
        for (size_t k = i; k < j; k++) lat_us_per_query.push_back(per_q_us);

        for (const auto& rs : rss) {
            for (const auto& r : rs) {
                checksum += r.prompt_cnt + r.cont_cnt;
                if (r.prompt_cnt > 0) hits++;
            }
        }
    }
    auto t_all1 = Clock::now();
    double total_ms = std::chrono::duration<double, std::milli>(t_all1 - t_all0).count();

    OpResult out;
    out.latency_us = summarize(std::move(lat_us_per_query));
    out.total_ms = total_ms;
    out.qps = static_cast<double>(queries.size()) / (total_ms / 1000.0);
    out.checksum = checksum;
    out.num_queries = queries.size();
    out.num_hits = hits;
    return out;
}

int main(int argc, char** argv) {
    BenchConfig cfg;
    if (!parse_args(argc, argv, &cfg)) {
        print_usage(argv[0]);
        return 2;
    }

    const std::string tokenized_path = cfg.index_dir + "/tokenized.0";
    const size_t total_tokens_in_file = get_total_tokens_in_file(tokenized_path);

    std::vector<U16> token_prefix;
    if (cfg.sample_tokens > 0) {
        const size_t prefix_tokens = std::min(cfg.sample_tokens, total_tokens_in_file);
        token_prefix = load_tokens_prefix(tokenized_path, prefix_tokens);
        if (token_prefix.size() < 128) {
            std::cerr << "token sample too small\n";
            return 2;
        }
    }

    std::cout << "CONFIG index_dir=" << cfg.index_dir
              << " sample_tokens=" << cfg.sample_tokens
              << " sampling_mode=" << (cfg.sample_tokens == 0 ? "full_file_random" : "prefix_random")
              << " total_tokens_in_file=" << total_tokens_in_file
              << " queries_per_len=" << cfg.queries_per_len
              << " lengths=" << join_lengths(cfg.lengths)
              << " warmup_n=" << cfg.warmup_n
              << " batch_size=" << cfg.batch_size
              << " runs=" << cfg.runs
              << " seed=" << cfg.seed
              << " hw_threads=" << std::thread::hardware_concurrency()
              << "\n";

    Engine<U16> engine({cfg.index_dir}, 2, 32000, 4, false, 0, 0, 0, {}, 512, false, {});

    size_t num_shards = engine.get_num_shards();
    U64 total_tok_cnt = 0;
    U64 total_ds_size = 0;
    for (size_t s = 0; s < num_shards; s++) {
        total_tok_cnt += engine.get_tok_cnt(s);
        total_ds_size += engine.get_ds_size(s);
    }
    std::cout << "INDEX_INFO num_shards=" << num_shards
              << " total_tokens=" << total_tok_cnt
              << " total_ds_bytes=" << total_ds_size
              << "\n";

    for (size_t len : cfg.lengths) {
        std::vector<std::vector<U16>> queries;
        if (cfg.sample_tokens == 0) {
            queries = build_fixed_len_queries_from_file(
                tokenized_path, total_tokens_in_file, cfg.queries_per_len, len, (U16)65535, cfg.seed + static_cast<uint32_t>(len * 7919));
        } else {
            queries = build_fixed_len_queries_from_prefix(
                token_prefix, cfg.queries_per_len, len, (U16)65535, cfg.seed + static_cast<uint32_t>(len * 7919));
        }
        if (queries.size() < cfg.queries_per_len) {
            std::cerr << "could not build enough queries for len=" << len
                      << " got=" << queries.size() << " want=" << cfg.queries_per_len << "\n";
            return 2;
        }

        std::unordered_set<uint64_t> unique_hashes;
        unique_hashes.reserve(queries.size() * 2);
        for (const auto& q : queries) unique_hashes.insert(hash_query(q));
        std::cout << "QUERYSET scale=len" << len
                  << " n=" << queries.size()
                  << " unique=" << unique_hashes.size()
                  << " unique_ratio=" << std::fixed << std::setprecision(4)
                  << (double)unique_hashes.size() / (double)queries.size()
                  << "\n";

        // warmup
        size_t warmup_n = std::min(cfg.warmup_n, queries.size());
        uint64_t warmup_checksum = 0;
        for (size_t i = 0; i < warmup_n; i++) {
            warmup_checksum += engine.count(queries[i]).count;
        }
        std::cout << "WARMUP scale=len" << len << " n=" << warmup_n << " checksum=" << warmup_checksum << "\n";

        for (size_t run = 1; run <= cfg.runs; run++) {
            std::string scale = "len" + std::to_string(len) + ":run" + std::to_string(run);
            print_op_result(scale, "count", bench_count(engine, queries));
            print_op_result(scale, "find", bench_find(engine, queries));
            print_op_result(scale, "prob", bench_prob(engine, queries));
            print_op_result(scale, "prob_batched", bench_prob_batched(engine, queries, cfg.batch_size));
            print_op_result(scale, "prob_sequence", bench_prob_sequence(engine, queries));
            print_op_result(scale, "prob_batched_sequence", bench_prob_batched_sequence(engine, queries, cfg.batch_size));
        }
    }

    return 0;
}
