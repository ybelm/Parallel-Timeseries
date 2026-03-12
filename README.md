# Parallel Time Series SAD Search

Sliding-window **Sum of Absolute Differences** search over time series data,
implemented in three versions: sequential, OpenMP (CPU), and CUDA (GPU).

Given a long time series T and a shorter query pattern P, the program finds
the position in T where P best matches, minimising:

```
SAD(T, P, i) = Σ |T[i+k] - P[k]|   for k = 0..m-1
```

---

## Project Structure

```
src/
├── timeseries.h / timeseries.cpp   # shared data structures, SAD kernel, benchmark
├── main_sequential.cpp             # sequential baseline
├── main_omp.cpp                    # OpenMP parallel implementation
└── main_cuda.cu                    # CUDA parallel implementation
```

---

## Requirements

| Tool | Version |
|------|---------|
| GCC  | ≥ 13    |
| NVCC | ≥ 12    |
| OpenMP | included with GCC |
| CUDA Toolkit | ≥ 12 |
| Thrust | included with CUDA Toolkit |

Optional: place `FordA_TRAIN.ts` (UCR Time Series Archive) in the same
directory as the executables to enable the real-data benchmarks.
Download: http://www.timeseriesclassification.com/aeon-toolkit/FordA.zip

---

## Build

```bash
# Build all three binaries
make all

# Build individually
make sequential
make omp
make cuda
```

---

## Run

```bash
# Sequential baseline
./sequential

# OpenMP (uses all available threads by default)
./omp

# CUDA (requires an NVIDIA GPU)
./main_cuda
```

Each binary runs a correctness test followed by a set of benchmarks
and prints results to stdout.

---

## Configuration

The main parameters are compile-time constants at the top of each `main_*.cpp`:

| Constant | Default | Description |
|----------|---------|-------------|
| `TS_LENGTH` | 500000 | Time series length |
| `PATTERN_LENGTH` | 5000 | Query pattern length |
| `N_RUNS` | 10 | Benchmark repetitions |

To change them, edit the constants and recompile with `make`.
