
CXX = g++
NVCC = nvcc

# Flags
CXXFLAGS = -std=c++17 -O3 -Wall -Wextra
OMPFLAGS = -fopenmp
VECFLAGS = -march=native -ftree-vectorize

#  File
SRC = src/timeseries.cpp
HDR = src/timeseries.h

SEQ_MAIN = src/main_sequential.cpp
OMP_MAIN = src/main_omp.cpp
CUDA_MAIN = src/main_cuda.cu

all: sequential omp cuda

sequential:
	$(CXX) $(CXXFLAGS) $(SRC) $(SEQ_MAIN) -o sequential
	@echo "Compiled: sequential"

omp:
	$(CXX) $(CXXFLAGS) $(VECFLAGS) $(OMPFLAGS) $(SRC) $(OMP_MAIN) -o omp
	@echo "Compiled: omp (OpenMP + vectorization)"

cuda:
	$(NVCC) -O2 -o main_cuda $(CUDA_MAIN) $(SRC) -Xcompiler -fopenmp
	@echo "Compiled: main_cuda (CUDA)"

# Run
run_seq: sequential
	./sequential

run_omp: omp
	./omp

# Clean
clean:
	rm -f sequential omp main_cuda

.PHONY: all sequential omp run_seq run_omp clean