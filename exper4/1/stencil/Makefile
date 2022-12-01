
CC = mpiicc 
OPT =
CFLAGS = -Wall -std=c99 $(OPT) -qopenmp -no-multibyte-chars -O2 -xHost -ffast-math -fma #-par-affinity=verbose
LDFLAGS = -Wall -qopenmp
LDLIBS = $(LDFLAGS)

targets = benchmark-mpi
objects = check.o benchmark.o stencil-mpi.o
# targets = benchmark-naive benchmark-omp benchmark-mpi benchmark-simd benchmark-block
# objects = check.o benchmark.o stencil-naive.o stencil-omp.o stencil-mpi.o stencil-simd.o stencil-block.o

.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

# benchmark-naive : check.o benchmark.o stencil-naive.o
# 	$(CC) -o $@ $^ $(LDLIBS)

# benchmark-omp : check.o benchmark.o stencil-omp.o
# 	$(CC) -o $@ $^ $(LDLIBS)

benchmark-mpi : check.o benchmark.o stencil-mpi.o
	$(CC) -o $@ $^ $(LDLIBS)
# benchmark-simd : check.o benchmark.o stencil-simd.o
# 	$(CC) -o $@ $^ $(LDLIBS)
# benchmark-block : check.o benchmark.o stencil-block.o
# 	$(CC) -o $@ $^ $(LDLIBS)


%.o : %.c common.h
	$(CC) -c $(CFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf $(targets) $(objects)