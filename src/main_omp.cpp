#include "timeseries.h"
#include <algorithm>
#include <cstdio>
#include <limits>
#include <omp.h>

/* Configuration */

static constexpr int TS_LENGTH = 500000;
static constexpr int PATTERN_LENGTH = 5000;
static constexpr int N_RUNS = 10;
static constexpr const char *UCR_TRAIN_PATH = "FordA_TRAIN.ts";

/* Parallel SAD */

SearchResult search_omp(const TimeSeries *ts, const TimeSeries *pattern,
                        omp_sched_t sched, int chunk_size, int n_threads)
{
    SearchResult result;
    if (pattern->data.size() > ts->data.size()) return result;

    size_t n_windows = ts->data.size() - pattern->data.size() + 1;
    size_t pat_len = pattern->data.size();

    float  best_sad = std::numeric_limits<float>::max();
    size_t best_pos = 0;

    omp_set_schedule(sched, chunk_size);

    #pragma omp parallel num_threads(n_threads)
    {
        float  local_best_sad = std::numeric_limits<float>::max();
        size_t local_best_pos = 0;

        #pragma omp for schedule(runtime) nowait
        for (size_t pos = 0; pos < n_windows; pos++) {
            float sad = sad_window(&ts->data[pos], pattern->data.data(), pat_len);
            if (sad < local_best_sad) {
                local_best_sad = sad;
                local_best_pos = pos;
            }
        }

        #pragma omp critical
        {
            if (local_best_sad < best_sad) {
                best_sad = local_best_sad;
                best_pos = local_best_pos;
            }
        }
    }

    result.best_sad = best_sad;
    result.best_pos = best_pos;
    return result;
}

/* Parallel multi-pattern search */
void search_multi_pattern_omp(const TimeSeries *ts, const TimeSeries *patterns,
                              SearchResult *results, size_t n_patterns, int n_threads)
{
    omp_set_max_active_levels(1);

    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (size_t p = 0; p < n_patterns; p++)
        results[p] = search_omp(ts, &patterns[p], omp_sched_static, 0, 1);
}

/* DB parallel search */
SearchResult search_db_omp(const TimeSeriesDB *db, const TimeSeries *pattern, int n_threads)
{
    SearchResult global;

    #pragma omp parallel for schedule(dynamic) num_threads(n_threads)
    for (size_t s = 0; s < db->count(); s++) {
        SearchResult local = search_sequential(&db->series[s], pattern);
        local.series_idx = s;

        #pragma omp critical
        {
            if (local.best_sad < global.best_sad)
                global = local;
        }
    }
    return global;
}

/* Benchmark wrapper */
struct OmpParams {
    const TimeSeries *ts;
    const TimeSeries *pattern;
    omp_sched_t sched;
    int chunk;
    int n_threads;
};

static OmpParams g_params;

static SearchResult search_omp_wrapper(const TimeSeries *ts, const TimeSeries *pattern) {
    return search_omp(ts, pattern, g_params.sched, g_params.chunk, g_params.n_threads);
}

/* Correctness test */

static int test_correctness(void) {
    std::printf("\n--- OMP correctness test ---\n");

    const size_t ts_len  = 5000;
    const size_t pat_len = 100;
    const size_t inject  = 1234;

    TimeSeries ts = ts_alloc(ts_len);
    TimeSeries pattern = ts_alloc(pat_len);
    TimeSeries src = ts_alloc(pat_len);

    ts_generate_random(&ts, 99);
    ts_generate_sine(&src, 5.0f, 0.0f);

    std::copy(src.data.begin(), src.data.end(), pattern.data.begin());
    std::copy(src.data.begin(), src.data.end(), ts.data.begin() + inject);

    int all_pass = 1;

    int thread_counts[] = { 1, 2, 4, 8 };
    omp_sched_t scheds[] = { omp_sched_static, omp_sched_dynamic,
                                    omp_sched_guided,  omp_sched_auto };
    const char *sched_names[] = { "static", "dynamic", "guided", "auto" };
    int n_scheds = (int)(sizeof(scheds) / sizeof(scheds[0]));
    int n_threads_n= (int)(sizeof(thread_counts) / sizeof(thread_counts[0]));

    for (int si = 0; si < n_scheds; si++) {
        for (int ti = 0; ti < n_threads_n; ti++) {
            SearchResult r = search_omp(&ts, &pattern, scheds[si], 0, thread_counts[ti]);
            int ok = (r.best_pos == inject && r.best_sad < 1e-4f);
            if (!ok) {
                std::printf("FAIL: sched=%-8s  threads=%d  pos=%zu  SAD=%.6f\n",
                            sched_names[si], thread_counts[ti], r.best_pos, r.best_sad);
                all_pass = 0;
            }
        }
    }
    std::printf("Correctness: %s\n", all_pass ? "PASS" : "FAIL");

    /* Multi-pattern correctness */
    const size_t N_PAT = 4;
    size_t inject_pos[N_PAT]  = { 100, 500, 1200, 2800 };
    TimeSeries mp_ts = ts_alloc(ts_len);
    TimeSeries mp_patterns[N_PAT];
    SearchResult mp_results [N_PAT];

    ts_generate_random(&mp_ts, 7);

    for (size_t p = 0; p < N_PAT; p++) {
        mp_patterns[p] = ts_alloc(pat_len);
        TimeSeries src_p = ts_alloc(pat_len);
        ts_generate_sine(&src_p, (float)(p + 1) * 1.5f, 0.0f);
        std::copy(src_p.data.begin(), src_p.data.end(), mp_patterns[p].data.begin());
        std::copy(src_p.data.begin(), src_p.data.end(), mp_ts.data.begin() + inject_pos[p]);
        ts_free(&src_p);
    }

    search_multi_pattern_omp(&mp_ts, mp_patterns, mp_results, N_PAT, 4);

    int mp_pass = 1;
    for (size_t p = 0; p < N_PAT; p++) {
        int ok = (mp_results[p].best_pos == inject_pos[p] && mp_results[p].best_sad < 1e-4f);
        if (!ok) {
            std::printf("FAIL: multi-pattern p=%zu  pos=%zu (expected %zu)  SAD=%.6f\n",
                        p, mp_results[p].best_pos, inject_pos[p], mp_results[p].best_sad);
            mp_pass = 0;
            all_pass = 0;
        }
        ts_free(&mp_patterns[p]);
    }
    ts_free(&mp_ts);
    std::printf("Multi-pattern correctness: %s\n", mp_pass ? "PASS" : "FAIL");

    ts_free(&ts);
    ts_free(&pattern);
    ts_free(&src);
    return all_pass ? 0 : 1;
}

/* Benchmark: thread scaling */

static void bench_thread_scaling(void) {
    std::printf("\n--- Thread scaling (ts=%d, pat=%d, sched=static) ---\n",
                TS_LENGTH, PATTERN_LENGTH);

    TimeSeries ts = ts_alloc(TS_LENGTH);
    TimeSeries pattern = ts_alloc(PATTERN_LENGTH);
    ts_generate_sine(&ts, 3.0f, 0.1f);
    ts_generate_sine(&pattern, 3.0f, 0.0f);

    BenchStats s_seq = benchmark(search_sequential, &ts, &pattern, N_RUNS);
    std::printf("sequential  mean=%8.4f s (baseline)\n", s_seq.mean);

    int thread_counts[] = { 1, 2, 4, 8, 16 };
    int n = (int)(sizeof(thread_counts) / sizeof(thread_counts[0]));

    for (int i = 0; i < n; i++) {
        g_params = { &ts, &pattern, omp_sched_static, 0, thread_counts[i] };
        BenchStats s = benchmark(search_omp_wrapper, &ts, &pattern, N_RUNS);
        std::printf("threads=%2d  mean=%8.4f s  speedup=%.2fx\n",
                    thread_counts[i], s.mean, s_seq.mean / s.mean);
    }

    ts_free(&ts);
    ts_free(&pattern);
}

/* Benchmark: scheduling strategies */

static void bench_scheduling(void) {
    std::printf("\n--- Scheduling strategies (ts=%d, pat=%d, threads=8) ---\n",
                TS_LENGTH, PATTERN_LENGTH);

    TimeSeries ts = ts_alloc(TS_LENGTH);
    TimeSeries pattern = ts_alloc(PATTERN_LENGTH);
    ts_generate_sine(&ts, 3.0f, 0.1f);
    ts_generate_sine(&pattern, 3.0f, 0.0f);

    BenchStats s_seq = benchmark(search_sequential, &ts, &pattern, N_RUNS);

    struct { omp_sched_t sched; int chunk; const char *label; } configs[] = {
        { omp_sched_static,  0,   "static  (auto)"    },
        { omp_sched_static,  1,   "static  chunk=1"   },
        { omp_sched_static,  8,   "static  chunk=8"   },
        { omp_sched_static,  64,  "static  chunk=64"  },
        { omp_sched_static,  512, "static  chunk=512" },
        { omp_sched_dynamic, 0,   "dynamic (auto)"    },
        { omp_sched_dynamic, 1,   "dynamic chunk=1"   },
        { omp_sched_dynamic, 8,   "dynamic chunk=8"   },
        { omp_sched_dynamic, 64,  "dynamic chunk=64"  },
        { omp_sched_guided,  0,   "guided  (auto)"    },
        { omp_sched_auto,    0,   "auto"              },
    };
    int n = (int)(sizeof(configs) / sizeof(configs[0]));

    for (int i = 0; i < n; i++) {
        g_params = { &ts, &pattern, configs[i].sched, configs[i].chunk, 8 };
        BenchStats s = benchmark(search_omp_wrapper, &ts, &pattern, N_RUNS);
        std::printf("%-22s  mean=%8.4f s  speedup=%.2fx\n",
                    configs[i].label, s.mean, s_seq.mean / s.mean);
    }

    ts_free(&ts);
    ts_free(&pattern);
}

/* Benchmark: sequential vs OMP (strong scaling) */
static void bench_seq_vs_omp(void) {
    std::printf("\n--- Sequential vs OMP (8 threads, static) ---\n");

    size_t lengths[] = { 10000, 50000, 100000, 250000, 500000 };
    size_t pat_len = 5000;
    int n = (int)(sizeof(lengths) / sizeof(lengths[0]));

    for (int i = 0; i < n; i++) {
        TimeSeries ts = ts_alloc(lengths[i]);
        TimeSeries pattern = ts_alloc(pat_len);
        ts_generate_sine(&ts, 2.0f, 0.1f);
        ts_generate_sine(&pattern, 2.0f, 0.0f);

        BenchStats s_seq = benchmark(search_sequential, &ts, &pattern, N_RUNS);

        g_params = { &ts, &pattern, omp_sched_static, 0, 8 };
        BenchStats s_omp = benchmark(search_omp_wrapper, &ts, &pattern, N_RUNS);

        std::printf("len=%7zu  seq=%8.4f s  omp=%8.4f s  speedup=%.2fx\n",
                    lengths[i], s_seq.mean, s_omp.mean, s_seq.mean / s_omp.mean);

        ts_free(&ts);
        ts_free(&pattern);
    }
}

/* Benchmark: multi-pattern */
/*
 * Strategy A - sequential loop over patterns, each search is sequential.
 * Strategy B - search_multi_pattern_omp, patterns distributed across threads.
 */
static void bench_multi_pattern(void) {
    std::printf("\n--- Multi-pattern search (ts=%d, pat=%d, 8 threads) ---\n",
                TS_LENGTH, PATTERN_LENGTH);

    size_t n_patterns_list[] = { 1, 2, 4, 8, 16 };
    int n = (int)(sizeof(n_patterns_list) / sizeof(n_patterns_list[0]));

    TimeSeries ts = ts_alloc(TS_LENGTH);
    ts_generate_sine(&ts, 3.0f, 0.1f);

    for (int i = 0; i < n; i++) {
        size_t N = n_patterns_list[i];

        std::vector<TimeSeries> patterns(N);
        std::vector<SearchResult> results (N);
        for (size_t p = 0; p < N; p++) {
            patterns[p] = ts_alloc(PATTERN_LENGTH);
            ts_generate_sine(&patterns[p], (float)(p + 1) * 0.5f, 0.0f);
        }

        // Strategy A: sequential
        double t0 = get_time_seconds();
        for (int r = 0; r < N_RUNS; r++)
            for (size_t p = 0; p < N; p++)
                results[p] = search_sequential(&ts, &patterns[p]);
        double t_seq = (get_time_seconds() - t0) / N_RUNS;

        // Strategy B: OMP multi-pattern
        double t1 = get_time_seconds();
        for (int r = 0; r < N_RUNS; r++)
            search_multi_pattern_omp(&ts, patterns.data(), results.data(), N, 8);
        double t_omp = (get_time_seconds() - t1) / N_RUNS;

        std::printf("n_patterns=%2zu  seq=%8.4f s  omp=%8.4f s  speedup=%.2fx\n",
                    N, t_seq, t_omp, t_omp > 0 ? t_seq / t_omp : 0.0);

        for (size_t p = 0; p < N; p++) ts_free(&patterns[p]);
    }

    ts_free(&ts);
}

/* Benchmark: real data (FordA) */

static void bench_real_data(void) {
    std::printf("\n--- Real data - FordA (len=%d, pat=%d, 8 threads, static) ---\n",
                TS_LENGTH, PATTERN_LENGTH);

    TimeSeries ts;
    if (!ts_load_ucr_concat(&ts, UCR_TRAIN_PATH, TS_LENGTH)) {
        std::printf("  [SKIP] %s not found.\n", UCR_TRAIN_PATH);
        return;
    }
    std::printf("  Loaded %zu points from %s\n", ts.data.size(), UCR_TRAIN_PATH);

    const size_t pat_offset = 1000;
    TimeSeries pattern = ts_alloc(PATTERN_LENGTH);
    std::copy(ts.data.begin() + pat_offset,
              ts.data.begin() + pat_offset + PATTERN_LENGTH,
              pattern.data.begin());

    BenchStats s_seq = benchmark(search_sequential, &ts, &pattern, N_RUNS);

    g_params = { &ts, &pattern, omp_sched_static, 0, 8 };
    BenchStats s_omp = benchmark(search_omp_wrapper, &ts, &pattern, N_RUNS);

    std::printf("seq=%8.4f s  omp=%8.4f s  speedup=%.2fx\n",
                s_seq.mean, s_omp.mean, s_seq.mean / s_omp.mean);

    ts_free(&ts);
    ts_free(&pattern);
}

static void print_summary() {
    std::printf("\n%s\n", std::string(65, '=').c_str());
    std::printf("  SUMMARY - single series (ts=%d, pat=%d)\n", TS_LENGTH, PATTERN_LENGTH);
    std::printf("%s\n", std::string(65, '=').c_str());
    std::printf("  %-12s  %10s  %10s  %10s\n", "Threads", "Mean (s)", "Min (s)", "Speedup");
    std::printf("  %s\n", std::string(48, '-').c_str());

    TimeSeries ts_ = ts_alloc(TS_LENGTH);
    TimeSeries pat_ = ts_alloc(PATTERN_LENGTH);
    ts_generate_sine(&ts_, 3.0f, 0.1f);
    ts_generate_sine(&pat_, 3.0f, 0.0f);

    BenchStats s_seq = benchmark(search_sequential, &ts_, &pat_, N_RUNS);
    std::printf("  %-12s  %10.4f  %10.4f  %10s\n", "sequential", s_seq.mean, s_seq.min, "1.00x");

    int thread_counts[] = { 1, 2, 4, 8, 16 };
    for (int t : thread_counts) {
        g_params = { &ts_, &pat_, omp_sched_static, 0, t };
        BenchStats s = benchmark(search_omp_wrapper, &ts_, &pat_, N_RUNS);
        char sp[16]; std::snprintf(sp, sizeof(sp), "%.2fx", s_seq.mean / s.mean);
        std::printf("  %-12d  %10.4f  %10.4f  %10s\n", t, s.mean, s.min, sp);
    }

    ts_free(&ts_); ts_free(&pat_);
    std::printf("%s\n", std::string(65, '=').c_str());
}

/* Main */

int main() {
    std::printf("--- Time Series SAD - OpenMP ---\n");
    std::printf("Max threads available: %d\n", omp_get_max_threads());
    std::printf("Dataset: ts_length=%d, pattern_length=%d\n\n", TS_LENGTH, PATTERN_LENGTH);

    if (test_correctness() != 0) {
        std::fprintf(stderr, "Correctness error.\n");
        return 1;
    }

    bench_thread_scaling();
    bench_scheduling();
    bench_seq_vs_omp();
    bench_multi_pattern();
    bench_real_data();

    std::printf("\nDone.\n");
    print_summary();
    return 0;
}
