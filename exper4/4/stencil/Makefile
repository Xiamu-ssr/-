
# MPICC = mpiicc
# MPICXX = mpiicpc

# OPT = -qopenmp -no-multibyte-chars -O2 -xHost -ffast-math -fma -ip -ipo#-par-affinity=verbose
# CFLAGS = -Wall -std=c99 $(OPT) 
# CXXFLAGS = -Wall -std=c++17 $(OPT)

# LDFLAGS = -Wall -qopenmp
# LDLIBS = $(LDFLAGS)

# targets = benchmark-omp
# objects = check.o benchmark.o stencil-omp.o
# # targets = benchmark-naive benchmark-omp benchmark-mpi benchmark-simd benchmark-block
# # objects = check.o benchmark.o stencil-naive.o stencil-omp.o stencil-mpi.o stencil-simd.o stencil-block.o

# .PHONY : default
# default : all

# .PHONY : all
# all : clean $(targets)


# benchmark-% : check.o benchmark.o stencil-%.o
# 	$(MPICXX) -o $@ $^ $(LDFLAGS) $(LDLIBS)
# # benchmark-naive : check.o benchmark.o stencil-naive.o
# # 	$(CC) -o $@ $^ $(LDLIBS)

# # benchmark-omp : check.o benchmark.o stencil-omp.o
# # 	$(CC) -o $@ $^ $(LDLIBS)
# check.o : check.c common.h
# 	$(MPICC) -c $(CFLAGS) $< -o $@
# benchmark.o : benchmark.c common.h
# 	$(MPICC) -c $(CFLAGS) $< -o $@
# # stencil-naive.o : stencil-naive.c common.h
# # 	$(MPICC) -c $(CFLAGS) $< -o $@

# %.o : %.cpp common.h
# 	$(MPICXX) -c $(CXXFLAGS) $< -o $@

# .PHONY: clean
# clean:
# 	rm -rf $(targets) $(objects)


MPICC = mpiicc
MPICXX = mpiicpc
OPT = -O2 -ffast-math -fast -xHost -ip -ipo #-par-affinity=verbose
FLAGS_COMMON = -Wall -g -qopenmp -no-multibyte-chars -DBT=$(BT) -DBZ=$(BZ) -DBY=$(BY) -DBX=$(BX)
CFLAGS = -std=c99 $(FLAGS_COMMON) $(OPT)
CXXFLAGS = -std=c++17 $(FLAGS_COMMON) $(OPT)
LDFLAGS = -Wall -qopenmp -g
LDLIBS = $(LDFLAGS)

targets =  $(proName)
objects = check.o benchmark.o stencil-omp.o 

.PHONY : default
default : all

.PHONY : all
all : $(targets)

$(proName) : check.o benchmark.o stencil-omp.o
	$(MPICXX) -o $@ $^ $(LDFLAGS) $(LDLIBS)

check.o : check.c common.h
	$(MPICC) -c $(CFLAGS) $< -o $@
benchmark.o : benchmark.c common.h
	$(MPICC) -c $(CFLAGS) $< -o $@
stencil-naive.o : stencil-naive.c common.h
	$(MPICC) -c $(CFLAGS) $< -o $@
%.o : %.cpp common.h
	$(MPICXX) -c $(CXXFLAGS) $< -o $@

.PHONY: clean
clean:
	rm -rf $(objects)
