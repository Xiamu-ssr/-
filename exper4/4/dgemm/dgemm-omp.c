#include <pmmintrin.h>
#include <immintrin.h>
#include <stdio.h>
#include <malloc.h>
#include <string.h>
#include <omp.h>
const char* dgemm_desc = "blocked 2 dgemm.";

#define ALIGNMENT 64
double A_cpy[1300 * 1300] __attribute__((aligned(ALIGNMENT)));
double B_cpy[1300 * 1300] __attribute__((aligned(ALIGNMENT)));
double C_cpy[1300 * 1300] __attribute__((aligned(ALIGNMENT)));
#define DOUBLE_PER_AVX512 8

#define UNROLLX 3
#define UNROLLY 8
//90112
#define BSI 96
#define BSJ 96
#define BSK 48
#define min(a,b) (((a)<(b))?(a):(b))

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* B, double* A, double* C, int threads_num){
    omp_set_max_active_levels(4);
    const int x = lda % (DOUBLE_PER_AVX512*UNROLLX);
    const int y = lda % UNROLLY;
    int X = lda + (x == 0 ? 0 : ((DOUBLE_PER_AVX512*UNROLLX) - x));
    int Y = lda + (y == 0 ? 0 : UNROLLY - y);
    memset(C_cpy, 0, X * Y * sizeof(double));

    #pragma omp parallel for num_threads(16)
    for (int i = 0; i < Y; ++i){
        for (int j = 0; j < X; ++j){
            if (i < lda && j < lda){
                A_cpy[j + X * i] = A[j + lda * i];
                B_cpy[j + X * i] = B[j + lda * i];
            }else{
                
                A_cpy[j + X * i] = 0;
                B_cpy[j + X * i] = 0;
            }
        }
    }
    #pragma omp parallel for num_threads(threads_num) //proc_bind(spread) schedule(guided,1)
    for (int j = 0; j < X; j += DOUBLE_PER_AVX512*UNROLLX){
		for (int i = 0; i < Y; i += UNROLLY){
			__m512d cij[UNROLLY][UNROLLX];
			for(int y=0; y<UNROLLY; ++y){
                for(int x=0; x<UNROLLX; ++x){
                    cij[y][x] = _mm512_load_pd(C_cpy+(i+y)*X+j+DOUBLE_PER_AVX512*x);
                }
            }
			for(int k=0; k<lda; ++k){
                __m512d bkj[UNROLLX];
                for(int x=0; x<UNROLLX; ++x){
                   bkj[x] = _mm512_load_pd(B_cpy+k*X+j+x*DOUBLE_PER_AVX512);
                }
                for(int y=0; y<UNROLLY; y+=2){
                    __m512d aik0 = _mm512_set1_pd(A_cpy[(i+y)*X+k]);
                    __m512d aik1 = _mm512_set1_pd(A_cpy[(i+y+1)*X+k]);
                    
                    for(int z=0; z<UNROLLX; ++z){
                        cij[y][z] = _mm512_fmadd_pd(aik0, bkj[z], cij[y][z]);
                        cij[y+1][z] = _mm512_fmadd_pd(aik1, bkj[z], cij[y+1][z]);
                    }
                }
            }
            for(int y=0; y<UNROLLY; ++y){
                for(int x=0; x<UNROLLX; ++x){
                    _mm512_store_pd(C_cpy+(i+y)*X+j+DOUBLE_PER_AVX512*x, cij[y][x]);
                }
            }
		}
    }

    #pragma omp parallel for num_threads(16)
    for(int i=0; i<lda; ++i){
        for(int j=0; j<lda; ++j){
            C[i*lda+j] += C_cpy[i*X+j];
        }
    }
}

