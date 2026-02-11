#include "timeseries.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <string>

/* Configuration */
static constexpr int TS_LENGTH = 500000;
static constexpr int PATTERN_LENGTH = 5000;
static constexpr int N_RUNS = 10;

/* Correctness test */
static int test_correctness(void) {
    std::printf("\n--- Correctness test ---\n");

    TimeSeries ts = ts_alloc(5000);
    TimeSeries pattern = ts_alloc(100);
    TimeSeries src = ts_alloc(100);

    ts_generate_random(&ts, 99);
    ts_generate_sine(&src, 5.0f, 0.0f);

    for (size_t i = 0; i < src.data.size(); i++) {
        pattern.data[i] = src.data[i];       // manual copy
        ts.data[i + 1234] = src.data[i];     // inject at fixed position
    }

    SearchResult r = search_sequential(&ts, &pattern);

    int ok = (r.best_pos == 1234 && r.best_sad < 1e-4f);
    std::printf("Result: %s\n", ok ? "PASS" : "FAIL");

    ts_free(&ts);
    ts_free(&pattern);
    ts_free(&src);
    return ok ? 0 : 1;
}

/* Benchmark: single series */
static void bench_single(void) {
    std::printf("--- Benchmark: single series ---\n");

    TimeSeries ts = ts_alloc(TS_LENGTH);
    TimeSeries pattern = ts_alloc(PATTERN_LENGTH);

    ts_generate_sine(&ts, 3.0f, 0.1f);
    ts_generate_sine(&pattern, 3.0f, 0.0f);

    BenchStats s = benchmark(search_sequential, &ts, &pattern, N_RUNS);
    stats_print(&s, "sequential");

    ts_free(&ts);
    ts_free(&pattern);
}

/* Benchmark: scaling only */
static void bench_scaling(void) {
    std::printf("\n--- Scaling benchmark ---\n");

    size_t lengths[] = { 10000, 50000, 100000, 500000 };
    for (size_t i = 0; i < sizeof(lengths)/sizeof(lengths[0]); i++) {
        TimeSeries ts = ts_alloc(lengths[i]);
        TimeSeries pattern = ts_alloc(PATTERN_LENGTH);

        ts_generate_sine(&ts, 2.0f, 0.1f);
        ts_generate_sine(&pattern, 2.0f, 0.0f);

        BenchStats s = benchmark(search_sequential, &ts, &pattern, N_RUNS);

        char label[64];
        std::sprintf(label, "ts_len=%zu", lengths[i]);
        stats_print(&s, label);

        ts_free(&ts);
        ts_free(&pattern);
    }
}

/* Main */
int main(void) {
    std::printf("=== Time Series SAD Sequential ===\n\n");

    int status = test_correctness();
    bench_single();
    bench_scaling();

    std::printf("\n%s\n", status == 0 ? "Done." : "Correctness test failed.");
    return status;
}