#include "timeseries.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

/* Configuration */

static constexpr int TS_LENGTH = 500000;
static constexpr int PATTERN_LENGTH = 5000;
static constexpr int N_RUNS = 10;

static constexpr const char *UCR_TRAIN_PATH = "FordA_TRAIN.ts";

/* Correctness test */
static int test_correctness(void) {
    std::printf("\n--- Correctness test ---\n");

    const size_t ts_len = 5000;
    const size_t pat_len = 100;
    const size_t inject = 1234;

    TimeSeries ts = ts_alloc(ts_len);
    TimeSeries pattern = ts_alloc(pat_len);
    TimeSeries src = ts_alloc(pat_len);

    ts_generate_random(&ts,  99);
    ts_generate_sine(&src, 5.0f, 0.0f);

    std::copy(src.data.begin(), src.data.end(), pattern.data.begin());
    std::copy(src.data.begin(), src.data.end(), ts.data.begin() + inject);

    SearchResult r = search_sequential(&ts, &pattern);
    result_print(&r, "correctness");

    int ok = (r.best_pos == inject && r.best_sad < 1e-4f);
    std::printf("Result: %s (expected pos=%zu, found pos=%zu, SAD=%.6f)\n\n",
                ok ? "PASS" : "FAIL", inject, r.best_pos, r.best_sad);

    ts_free(&ts);
    ts_free(&pattern);
    ts_free(&src);
    return ok ? 0 : 1;
}

/* Benchmark: single series (synthetic) */
static void bench_single(void) {
    std::printf("--- Benchmark: single series (len=%d, pat=%d) ---\n", TS_LENGTH, PATTERN_LENGTH);

    TimeSeries ts = ts_alloc(TS_LENGTH);
    TimeSeries pattern = ts_alloc(PATTERN_LENGTH);

    ts_generate_sine(&ts, 3.0f, 0.1f);
    ts_generate_sine(&pattern, 3.0f, 0.0f);

    BenchStats s = benchmark(search_sequential, &ts, &pattern, N_RUNS);
    stats_print (&s, "sequential");

    SearchResult r = search_sequential(&ts, &pattern);
    result_print(&r, "best match");

    ts_free(&ts);
    ts_free(&pattern);
}

/* Benchmark: real data (FordA) */
/*
 * Loads FordA_TRAIN.ts and concatenates all series into a single flat
 * TimeSeries of length TS_LENGTH. The pattern is taken as the first
 * PATTERN_LENGTH points of the series itself (a real sub-sequence).
 *
 * If the file is not present the benchmark is skipped.
 */
static void bench_real_data(void) {
    std::printf("\n--- Benchmark: real data - FordA (len=%d, pat=%d) ---\n", TS_LENGTH, PATTERN_LENGTH);

    TimeSeries ts;
    if (!ts_load_ucr_concat(&ts, UCR_TRAIN_PATH, TS_LENGTH)) {
        std::printf("  [SKIP] %s not found - place the file next to the executable.\n", UCR_TRAIN_PATH);
        return;
    }
    std::printf("  Loaded %zu points from %s\n", ts.data.size(), UCR_TRAIN_PATH);

    // Pattern: extract a real sub-sequence from a known offset
    const size_t pat_offset = 1000;
    TimeSeries pattern = ts_alloc(PATTERN_LENGTH);
    std::copy(ts.data.begin() + pat_offset, ts.data.begin() + pat_offset + PATTERN_LENGTH, pattern.data.begin());
    pattern.label = "forda_subseq";

    BenchStats s = benchmark(search_sequential, &ts, &pattern, N_RUNS);
    stats_print (&s, "sequential (real)");

    SearchResult r = search_sequential(&ts, &pattern);
    result_print(&r, "best match (real)");
    // the injected offset must be the best (or tied) match
    std::printf("  Injected at pos=%zu, found pos=%zu  %s\n",
                pat_offset, r.best_pos,
                r.best_pos == pat_offset ? "(exact)" : "(overlap allowed)");

    ts_free(&ts);
    ts_free(&pattern);
}

/* Benchmark: scaling */
static void bench_scaling(void) {
    std::printf("\n--- Scaling: series size effect ---\n");

    size_t lengths[] = { 10000, 50000, 100000, 250000, 500000 };
    size_t pat_len = 5000;
    int n = (int)(sizeof(lengths) / sizeof(lengths[0]));

    for (int i = 0; i < n; i++) {
        TimeSeries ts = ts_alloc(lengths[i]);
        TimeSeries pattern = ts_alloc(pat_len);

        ts_generate_sine(&ts, 2.0f, 0.1f);
        ts_generate_sine(&pattern, 2.0f, 0.0f);

        BenchStats s = benchmark(search_sequential, &ts, &pattern, N_RUNS);

        char label[64];
        std::snprintf(label, sizeof(label), "ts_len=%6zu", lengths[i]);
        stats_print(&s, label);

        ts_free(&ts);
        ts_free(&pattern);
    }
}

/* Main */
int main(void) {
    std::printf("---------------------------------------------------\n");
    std::printf("=  Time Series SAD Search - Sequential Version    =\n");
    std::printf("---------------------------------------------------\n\n");

    int status = test_correctness();
    bench_single();
    bench_real_data();
    bench_scaling();

    std::printf("\n%s\n", status == 0 ? "Done." : "Correctness test failed.");

    
    std::printf("\n%-45s  %8s\n", "Benchmark", "Mean (s)");
    std::printf("%-45s  %8s\n", std::string(45, '-').c_str(), "--------");
    std::printf("\nDone (sequential baseline - no speedup to report).\n");

    return status;
}
