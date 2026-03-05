#include "timeseries.h"

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <limits>
#include <cuda_runtime.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>

/* Configuration */
static constexpr int TS_LENGTH = 500000;
static constexpr int PATTERN_LENGTH = 5000;
static constexpr int N_RUNS = 10;
static constexpr const char *UCR_TRAIN_PATH = "FordA_TRAIN.ts";

/*CUDA error checking */
#define CUDA_CHECK(call)                                                       \
    do {                                                                       \
        cudaError_t err = (call);                                              \
        if (err != cudaSuccess) {                                              \
            std::fprintf(stderr, "[CUDA ERROR] %s:%d  %s\n",                  \
                         __FILE__, __LINE__, cudaGetErrorString(err));         \
            std::exit(EXIT_FAILURE);                                           \
        }                                                                      \
    } while (0)

/* Kernel: global memory */
/*
 * Each thread computes the SAD of one window.
 */
__global__ void kernel_sad(const float *d_ts, const float *d_pattern,
                           float *d_sad, size_t n_windows, size_t pat_len)
{
    size_t pos = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n_windows) return;

    float sad = 0.0f;
    for (size_t i = 0; i < pat_len; ++i)
        sad += fabsf(d_ts[pos + i] - d_pattern[i]);
    d_sad[pos] = sad;
}

/* Kernel: shared memory */
/*
 * The pattern is loaded cooperatively into shared memory to reduce
 * redundant global reads across threads in the same block.
 */
__global__ void kernel_sad_shared(const float *d_ts, const float *d_pattern,
                                  float *d_sad, size_t n_windows, size_t pat_len)
{
    extern __shared__ float s_pattern[];

    for (size_t i = threadIdx.x; i < pat_len; i += blockDim.x)
        s_pattern[i] = d_pattern[i];
    __syncthreads();

    size_t pos = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pos >= n_windows) return;

    float sad = 0.0f;
    for (size_t i = 0; i < pat_len; ++i)
        sad += fabsf(d_ts[pos + i] - s_pattern[i]);
    d_sad[pos] = sad;
}

/* CUDA launch parameters */

struct CudaParams {
    int  block_size;
    bool use_shared_mem;
};

/* Device context */
/*
 * GPU buffers are allocated once and reused across benchmark runs so that
 * the measured time covers kernel + Thrust reduction only, not allocation
 * and H2D transfer.
 *
 * Lifecycle:
 *   cuda_context_init()  - allocate + upload (once per series/pattern pair)
 *   search_cuda_ctx()    - kernel + reduce (called N_RUNS times)
 *   cuda_context_free()  - release device memory
 */
struct CudaContext {
    float *d_ts = nullptr;
    float *d_pattern = nullptr;
    float *d_sad = nullptr;
    size_t ts_len = 0;
    size_t pat_len = 0;
    size_t n_windows = 0;
};

static CudaContext cuda_context_init(const TimeSeries *ts, const TimeSeries *pattern)
{
    CudaContext ctx;
    ctx.ts_len = ts->data.size();
    ctx.pat_len = pattern->data.size();
    ctx.n_windows = ctx.ts_len - ctx.pat_len + 1;

    CUDA_CHECK(cudaMalloc(&ctx.d_ts, ctx.ts_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_pattern, ctx.pat_len * sizeof(float)));
    CUDA_CHECK(cudaMalloc(&ctx.d_sad, ctx.n_windows * sizeof(float)));

    CUDA_CHECK(cudaMemcpy(ctx.d_ts, ts->data.data(), ctx.ts_len * sizeof(float), cudaMemcpyHostToDevice));
    CUDA_CHECK(cudaMemcpy(ctx.d_pattern, pattern->data.data(), ctx.pat_len * sizeof(float), cudaMemcpyHostToDevice));
    return ctx;
}

static void cuda_context_free(CudaContext *ctx)
{
    CUDA_CHECK(cudaFree(ctx->d_ts));
    CUDA_CHECK(cudaFree(ctx->d_pattern));
    CUDA_CHECK(cudaFree(ctx->d_sad));
    *ctx = CudaContext{};
}

/* Core search (uses pre-allocated context) */
static SearchResult search_cuda_ctx(const CudaContext *ctx, const CudaParams *params)
{
    SearchResult result;

    int block_size = params->block_size;
    int grid_size  = (int)((ctx->n_windows + block_size - 1) / block_size);

    if (params->use_shared_mem) {
        size_t shared_bytes = ctx->pat_len * sizeof(float);

        int shared_limit_int = 0;
        CUDA_CHECK(cudaDeviceGetAttribute(&shared_limit_int, cudaDevAttrMaxSharedMemoryPerBlock, 0));
        size_t shared_limit = static_cast<size_t>(shared_limit_int);

        if (shared_bytes <= shared_limit) {
            kernel_sad_shared<<<grid_size, block_size, shared_bytes>>>(
                ctx->d_ts, ctx->d_pattern, ctx->d_sad,
                ctx->n_windows, ctx->pat_len);
        } else {
            std::printf("[WARN] pat_len=%zu exceeds shared mem (%zu B), "
                        "falling back to global\n", ctx->pat_len, shared_limit);
            kernel_sad<<<grid_size, block_size>>>(
                ctx->d_ts, ctx->d_pattern, ctx->d_sad,
                ctx->n_windows, ctx->pat_len);
        }
    } else {
        kernel_sad<<<grid_size, block_size>>>(
            ctx->d_ts, ctx->d_pattern, ctx->d_sad,
            ctx->n_windows, ctx->pat_len);
    }

    CUDA_CHECK(cudaGetLastError());
    CUDA_CHECK(cudaDeviceSynchronize());

    // GPU reduction via Thrust: avoids copying all n_windows floats to host
    thrust::device_ptr<float> d_ptr(ctx->d_sad);
    thrust::device_ptr<float> d_min_ptr =
        thrust::min_element(d_ptr, d_ptr + ctx->n_windows);

    result.best_pos = static_cast<size_t>(d_min_ptr - d_ptr);
    CUDA_CHECK(cudaMemcpy(&result.best_sad, thrust::raw_pointer_cast(d_min_ptr),
                          sizeof(float), cudaMemcpyDeviceToHost));
    return result;
}

static SearchResult search_cuda(const TimeSeries *ts, const TimeSeries *pattern, const CudaParams *params)
{
    if (!ts || !pattern || pattern->data.size() > ts->data.size())
        return SearchResult{};

    CudaContext ctx = cuda_context_init(ts, pattern);
    SearchResult r = search_cuda_ctx(&ctx, params);
    cuda_context_free(&ctx);
    return r;
}

/* Benchmark wrapper factory */
/*
 * Lambda captures a pre-built CudaContext by pointer so that cudaMalloc/
 * cudaFree are not called on every benchmark run.
 */
static auto make_cuda_wrapper(const CudaContext *ctx, const CudaParams *params)
{
    return [=](const TimeSeries *, const TimeSeries *) -> SearchResult {
        return search_cuda_ctx(ctx, params);
    };
}

/* GPU info */
static void print_gpu_info()
{
    int device_count = 0;
    CUDA_CHECK(cudaGetDeviceCount(&device_count));
    std::printf("GPUs detected: %d\n", device_count);

    for (int i = 0; i < device_count; ++i) {
        cudaDeviceProp prop;
        CUDA_CHECK(cudaGetDeviceProperties(&prop, i));
        std::printf("  [GPU %d] %s  |  SM: %d  |  VRAM: %.0f MB  |  " "Max threads/block: %d  |  Shared mem/block: %zu KB\n",
                    i, prop.name, prop.multiProcessorCount, (double)prop.totalGlobalMem / (1024.0 * 1024.0),
                    prop.maxThreadsPerBlock, prop.sharedMemPerBlock / 1024);
    }
}

/* Correctness test */
static int test_correctness()
{
    std::printf("\n--- CUDA correctness test ---\n");

    constexpr size_t ts_len = 5000;
    constexpr size_t pat_len = 100;
    constexpr size_t inject = 1234;

    TimeSeries ts = ts_alloc(ts_len);
    TimeSeries pattern = ts_alloc(pat_len);
    TimeSeries src = ts_alloc(pat_len);

    ts_generate_random(&ts, 99);
    ts_generate_sine(&src, 5.0f, 0.0f);

    std::copy(src.data.begin(), src.data.end(), pattern.data.begin());
    std::copy(src.data.begin(), src.data.end(), ts.data.begin() + inject);

    int all_pass = 1;

    int  block_sizes[] = { 128, 256, 512 };
    bool shared_modes[] = { false, true };
    const char *sm_names[]= { "global", "shared" };

    for (int bi = 0; bi < 3; ++bi) {
        for (int si = 0; si < 2; ++si) {
            CudaParams p = { block_sizes[bi], shared_modes[si] };
            SearchResult r = search_cuda(&ts, &pattern, &p);
            int ok = (r.best_pos == inject && r.best_sad < 1e-4f);
            if (!ok) {
                std::printf("FAIL  block=%3d  mem=%-6s  pos=%zu  SAD=%.6f\n",
                            block_sizes[bi], sm_names[si], r.best_pos, r.best_sad);
                all_pass = 0;
            }
        }
    }

    std::printf("Correctness: %s\n", all_pass ? "PASS (all configurations)" : "FAIL");

    ts_free(&ts);
    ts_free(&pattern);
    ts_free(&src);
    return all_pass ? 0 : 1;
}


/* Benchmark: CUDA vs Sequential */
static void bench_cuda_vs_seq()
{
    std::printf("\n--- Sequential vs CUDA (block=256, shared) ---\n");

    size_t lengths[] = { 10000, 50000, 100000, 250000, 500000 };
    constexpr size_t pat_len = 5000;
    int n = (int)(sizeof(lengths) / sizeof(lengths[0]));

    for (int i = 0; i < n; ++i) {
        TimeSeries ts = ts_alloc(lengths[i]);
        TimeSeries pattern = ts_alloc(pat_len);
        ts_generate_sine(&ts, 2.0f, 0.1f);
        ts_generate_sine(&pattern, 2.0f, 0.0f);

        BenchStats s_seq = benchmark(search_sequential, &ts, &pattern, N_RUNS);

        CudaContext ctx = cuda_context_init(&ts, &pattern);
        CudaParams params = { 256, true };
        auto fn = make_cuda_wrapper(&ctx, &params);
        BenchStats s_cuda = benchmark(fn, &ts, &pattern, N_RUNS);
        cuda_context_free(&ctx);

        std::printf("len=%6zu  seq=%.4f ms  cuda=%.4f ms  speedup=%.2fx\n",
                    lengths[i], s_seq.mean  * 1e3, s_cuda.mean * 1e3, s_seq.mean  / s_cuda.mean);

        ts_free(&ts);
        ts_free(&pattern);
    }
}

/* Benchmark: real data (FordA) */
static void bench_real_data()
{
    std::printf("\n--- Real data - FordA (len=%d, pat=%d, block=256, shared) ---\n", TS_LENGTH, PATTERN_LENGTH);

    TimeSeries ts;
    if (!ts_load_ucr_concat(&ts, UCR_TRAIN_PATH, TS_LENGTH)) {
        std::printf("  [SKIP] %s not found.\n", UCR_TRAIN_PATH);
        return;
    }
    std::printf("  Loaded %zu points from %s\n", ts.data.size(), UCR_TRAIN_PATH);

    const size_t pat_offset = 1000;
    TimeSeries pattern = ts_alloc(PATTERN_LENGTH);
    std::copy(ts.data.begin() + pat_offset, ts.data.begin() + pat_offset + PATTERN_LENGTH,
              pattern.data.begin());

    BenchStats s_seq = benchmark(search_sequential, &ts, &pattern, N_RUNS);
    CudaContext ctx = cuda_context_init(&ts, &pattern);
    CudaParams params = { 256, true };
    auto fn = make_cuda_wrapper(&ctx, &params);
    BenchStats s_cuda = benchmark(fn, &ts, &pattern, N_RUNS);
    cuda_context_free(&ctx);

    std::printf("seq=%.4f ms  cuda=%.4f ms  speedup=%.2fx\n",
                s_seq.mean * 1e3, s_cuda.mean * 1e3, s_seq.mean  / s_cuda.mean);

    ts_free(&ts);
    ts_free(&pattern);
}

/* Main */
int main()
{
    std::printf("--- Time Series SAD - CUDA ---\n");
    print_gpu_info();

    if (test_correctness() != 0) {
        std::fprintf(stderr, "Correctness error - aborting.\n");
        return 1;
    }

    bench_cuda_vs_seq();
    bench_real_data();

    std::printf("\nDone.\n");
    return 0;
}
