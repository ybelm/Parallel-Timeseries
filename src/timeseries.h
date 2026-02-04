#ifndef TIMESERIES_H
#define TIMESERIES_H

#include <cstddef>
#include <cstdio>
#include <limits>
#include <string>
#include <vector>
#include <cmath>


/* Data structures */
struct TimeSeries {
    std::vector<float> data;
    std::string label;

    size_t length() const { return data.size(); }
};

struct TimeSeriesDB {
    std::vector<TimeSeries> series;

    size_t count() const { return series.size(); }
};

struct SearchResult {
    size_t best_pos = 0;
    float  best_sad = std::numeric_limits<float>::max();
    size_t series_idx = 0;
};

struct BenchStats {
    double mean = 0.0;
    double std = 0.0;
    double min = 0.0;
    double max = 0.0;
    int n_runs = 0;
};

/* Allocation */
TimeSeries ts_alloc(size_t length);
void ts_free(TimeSeries *ts);

TimeSeriesDB db_alloc(size_t count, size_t series_length);
void db_free(TimeSeriesDB *db);

/* Synthetic data generation */
void ts_generate_sine(TimeSeries *ts, float freq, float noise_level);
void ts_generate_random(TimeSeries *ts, unsigned int seed);

/* SAD */

float sad_window(const float *window, const float *pattern, size_t len);

/* Sequential search */

SearchResult search_sequential(const TimeSeries *ts, const TimeSeries *pattern);
SearchResult search_db_sequential(const TimeSeriesDB *db,const TimeSeries *pattern);

/* Timer */
double get_time_seconds(void);

/* Benchmark */
template <typename SearchFn>
BenchStats benchmark(SearchFn search_fn, const TimeSeries *ts, const TimeSeries *pattern, int n_runs)
{
    std::vector<double> times;
    times.reserve(static_cast<size_t>(n_runs));

    // warm-up
    (void)search_fn(ts, pattern);
    (void)search_fn(ts, pattern);

    for (int r = 0; r < n_runs; ++r) {
        double t0 = get_time_seconds();
        (void)search_fn(ts, pattern);
        double t1 = get_time_seconds();
        times.push_back(t1 - t0);
    }

    BenchStats s;
    s.n_runs = n_runs;
    s.min = std::numeric_limits<double>::max();
    s.max = 0.0;

    double sum = 0.0, sum_sq = 0.0;
    for (double t : times) {
        sum += t;
        sum_sq += t * t;
        if (t < s.min) s.min = t;
        if (t > s.max) s.max = t;
    }

    s.mean = sum / n_runs;
    double var = sum_sq / n_runs - s.mean * s.mean;
    if (var < 0.0) var = 0.0;
    s.std = std::sqrt(var);
    return s;
}

template <typename SearchFn>
BenchStats benchmark_db(SearchFn search_fn, const TimeSeriesDB *db, const TimeSeries *pattern, int n_runs)
{
    std::vector<double> times;
    times.reserve(static_cast<size_t>(n_runs));

    (void)search_fn(db, pattern);
    (void)search_fn(db, pattern);

    for (int r = 0; r < n_runs; ++r) {
        double t0 = get_time_seconds();
        (void)search_fn(db, pattern);
        double t1 = get_time_seconds();
        times.push_back(t1 - t0);
    }

    BenchStats s;
    s.n_runs = n_runs;
    s.min = std::numeric_limits<double>::max();
    s.max = 0.0;

    double sum = 0.0, sum_sq = 0.0;
    for (double t : times) {
        sum += t;
        sum_sq += t * t;
        if (t < s.min) s.min = t;
        if (t > s.max) s.max = t;
    }

    s.mean = sum / n_runs;
    double var = sum_sq / n_runs - s.mean * s.mean;
    if (var < 0.0) var = 0.0;
    s.std = std::sqrt(var);
    return s;
}

/* Utilities */
void stats_print(const BenchStats *s, const char *label);
void result_print(const SearchResult *r, const char *label);
#endif
