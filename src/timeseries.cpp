#include "timeseries.h"

#include <cmath>
#include <chrono>
#include <fstream>
#include <iomanip>
#include <random>
#include <sstream>

/* Allocation */
TimeSeries ts_alloc(size_t length) {
    TimeSeries ts;
    ts.data.resize(length);
    ts.label.clear();
    return ts;
}

void ts_free(TimeSeries *ts) {
    if (ts) {
        ts->data.clear();
        ts->data.shrink_to_fit();
        ts->label.clear();
    }
}

TimeSeriesDB db_alloc(size_t count, size_t series_length) {
    TimeSeriesDB db;
    db.series.resize(count);
    for (size_t i = 0; i < count; ++i)
        db.series[i] = ts_alloc(series_length);
    return db;
}

void db_free(TimeSeriesDB *db) {
    if (db) {
        for (auto &ts : db->series) ts_free(&ts);
        db->series.clear();
        db->series.shrink_to_fit();
    }
}

/* Synthetic data generation */
void ts_generate_sine(TimeSeries *ts, float freq, float noise_level) {
    if (!ts) return;

    std::mt19937 rng(42);
    std::uniform_real_distribution<float> noise_dist(-1.0f, 1.0f);
    const float two_pi = 2.0f * std::acos(-1.0f);

    for (size_t i = 0; i < ts->data.size(); ++i) {
        float t = static_cast<float>(i) / static_cast<float>(ts->data.size());
        float signal = std::sin(two_pi * freq * t);
        float noise = noise_level * noise_dist(rng);
        ts->data[i] = signal + noise;
    }

    std::ostringstream oss;
    oss << "sine_f" << std::fixed << std::setprecision(2) << freq
        << "_n" << std::fixed << std::setprecision(2) << noise_level;
    ts->label = oss.str();
}

void ts_generate_random(TimeSeries *ts, unsigned int seed) {
    if (!ts) return;

    std::mt19937 rng(seed);
    std::uniform_real_distribution<float> dist(-1.0f, 1.0f);
    for (auto &x : ts->data) x = dist(rng);

    std::ostringstream oss;
    oss << "random_seed" << seed;
    ts->label = oss.str();
}

/* Real data loading */

/*
 * UCR .ts format:
 *
 * Strategy: read every data line, strip the label prefix (up to the first
 * comma if present), parse the floats, and append them all into a single
 * flat vector.  Finally, truncate to target_length so the output is always
 * exactly the size the caller requested.
 */
bool ts_load_ucr_concat(TimeSeries *ts, const char *filepath, size_t target_length)
{
    std::ifstream f(filepath);
    if (!f.is_open()) {
        std::fprintf(stderr, "[UCR] Cannot open: %s\n", filepath);
        return false;
    }

    ts->data.clear();
    ts->data.reserve(target_length);

    std::string line;
    while (std::getline(f, line) && ts->data.size() < target_length) {
        if (line.empty() || line[0] == '@') continue;   // skip header

        // Strip class label: everything up to and including the first comma
        size_t comma = line.find(',');
        std::string values = (comma != std::string::npos) ? line.substr(comma + 1) : line;

        std::istringstream ss(values);
        float val;
        while (ss >> val && ts->data.size() < target_length)
            ts->data.push_back(val);
    }

    if (ts->data.empty()) {
        std::fprintf(stderr, "[UCR] No data found in: %s\n", filepath);
        return false;
    }

    // Truncate to exactly target_length
    ts->data.resize(target_length);
    ts->label = "ucr_forda";
    return true;
}


/* SAD */
float sad_window(const float *window, const float *pattern, size_t len) {
    float sad = 0.0f;

    #if defined(_OPENMP) && !defined(__CUDACC__)
    #pragma omp simd reduction(+:sad)
    #endif
    for (size_t i = 0; i < len; ++i)
        sad += std::fabs(window[i] - pattern[i]);
    return sad;
}

/* Sequential search */
SearchResult search_sequential(const TimeSeries *ts, const TimeSeries *pattern) {
    SearchResult result;
    if (!ts || !pattern) return result;
    if (pattern->data.size() > ts->data.size()) return result;

    size_t n_windows = ts->data.size() - pattern->data.size() + 1;
    for (size_t pos = 0; pos < n_windows; ++pos) {
        float sad = sad_window(&ts->data[pos], pattern->data.data(), pattern->data.size());
        if (sad < result.best_sad) {
            result.best_sad = sad;
            result.best_pos = pos;
        }
    }
    return result;
}

SearchResult search_db_sequential(const TimeSeriesDB *db, const TimeSeries *pattern) {
    SearchResult best;
    if (!db || !pattern) return best;

    for (size_t s = 0; s < db->series.size(); ++s) {
        SearchResult r = search_sequential(&db->series[s], pattern);
        if (r.best_sad < best.best_sad) {
            best = r;
            best.series_idx = s;
        }
    }
    return best;
}

/* Timer */

double get_time_seconds() {
    using clock = std::chrono::steady_clock;
    return std::chrono::duration<double>(clock::now().time_since_epoch()).count();
}

/* Print utilities */
void stats_print(const BenchStats *s, const char *label) {
    if (!s) return;
    std::printf("  %-28s  mean=%8.4f s  std=%7.4f s  min=%8.4f s  max=%8.4f s  (n=%d)\n",
                label ? label : "benchmark",
                s->mean, s->std, s->min, s->max, s->n_runs);
}

void result_print(const SearchResult *r, const char *label) {
    if (!r) return;
    std::printf("[%s]  best_pos=%zu  best_SAD=%.4f  series_idx=%zu\n",
                label ? label : "result",
                r->best_pos, r->best_sad, r->series_idx);
}
