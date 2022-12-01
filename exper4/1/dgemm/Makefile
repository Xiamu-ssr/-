# We will benchmark you against Intel MKL implementation, the default processor vendor-tuned implementation.
# This makefile is intended for the Intel C compiler.
# Your code must compile (with icc) with the given CFLAGS. You may experiment with the OPT variable to invoke additional compiler options.

CC = icc
OPT = -no-multibyte-chars -xHost -funroll-loops -mfma -ffast-math -qopenmp
CFLAGS = -Wall -DGETTIMEOFDAY -std=c99 $(OPT)
LDFLAGS = -Wall 
# mkl is needed for blas implementation
LDLIBS = -lmkl_intel_lp64 -lmkl_sequential -lmkl_core -lpthread -lm -qopenmp

# targets = benchmark-test benchmark-naive benchmark-blocked benchmark-blas benchmark-blocked-1 benchmark-blocked-2
# objects = benchmark-test.o benchmark.o dgemm-naive.o dgemm-blocked.o dgemm-blas.o dgemm-blocked-1.o dgemm-blocked-2.o
targets = benchmark-omp
objects = benchmark.o dgemm-omp.o
 
.PHONY : default
default : all

.PHONY : all
all : clean $(targets)

# benchmark-test : benchmark-test.o dgemm-blocked.o
# 	$(CC) -o $@ $^ $(LDLIBS)

# benchmark-naive : benchmark.o dgemm-naive.o
# 	$(CC) -o $@ $^ $(LDLIBS)
# benchmark-blocked : benchmark.o dgemm-blocked.o
# 	$(CC) -o $@ $^ $(LDLIBS)
# benchmark-blas : benchmark.o dgemm-blas.o
# 	$(CC) -o $@ $^ $(LDLIBS)
# benchmark-blocked-1 : benchmark.o dgemm-blocked-1.o
# 	$(CC) -o $@ $^ $(LDLIBS)
benchmark-omp : benchmark.o dgemm-omp.o
	$(CC) -o $@ $^ $(LDLIBS)
# benchmark-blocked-2 : benchmark.o dgemm-blocked-2.o
# 	$(CC) -o $@ $^ $(LDLIBS)


%.o : %.c
	$(CC) -c $(CFLAGS) $<

.PHONY : clean
clean:
	rm -f $(targets) $(objects)
