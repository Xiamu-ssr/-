#include <pmmintrin.h>
#include<immintrin.h>
#include<stdio.h>
const char* dgemm_desc = "blocked 2 dgemm.";

#define BSI 64
#define BSJ 64
#define BSK 64
#define DOUBLE_PER_AVX512 8
#define DOUBLE_PER_AVX256 4

#define UNROLL 4
#define min(a,b) (((a)<(b))?(a):(b))

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static void do_block (int lda, int I, int J, int K, double* A, double* B, double* C){
    int top = J - J%(DOUBLE_PER_AVX512*UNROLL);
    for(int j=0; j<top; j+=DOUBLE_PER_AVX512*UNROLL){
        for(int i=0; i<I; ++i){
            __m512d cij[UNROLL];
            for(int x=0; x<UNROLL; ++x){
                cij[x] = _mm512_load_pd(C+i*lda+j+DOUBLE_PER_AVX512*x);
            }
            for(int k=0; k<K; ++k){
                __m512d aik = _mm512_broadcastsd_pd(_mm_loaddup_pd(A+i*lda+k));
                for(int x=0; x<UNROLL; ++x){
                    cij[x] = _mm512_add_pd(cij[x], _mm512_mul_pd(_mm512_load_pd(B+k*lda+j+DOUBLE_PER_AVX512*x), aik));
                }
            }
            for(int x=0; x<UNROLL; ++x){
                _mm512_store_pd(C+i*lda+j+DOUBLE_PER_AVX512*x, cij[x]);
            }
        }
    }
    int top2 = J - J%DOUBLE_PER_AVX512;
    if(J-top>=8){
        for(int j=top; j<top2; j+=DOUBLE_PER_AVX512){
            for(int i=0; i<I; ++i){
                __m512d cij = _mm512_load_pd(C+i*lda+j);
                for(int k=0; k<K; ++k){
                    __m512d aik = _mm512_broadcastsd_pd(_mm_loaddup_pd(A+i*lda+k));
                    cij = _mm512_add_pd(cij, _mm512_mul_pd(_mm512_load_pd(B+k*lda+j), aik));
                }
                _mm512_store_pd(C+i*lda+j,cij);
            }
        }
    }
    
    if(J-top2>=4){
        for(int j=top2; j<top2+4; j+=DOUBLE_PER_AVX256){
            for(int i=0; i<I; ++i){
                __m256d cij = _mm256_load_pd(C+i*lda+j);
                for(int k=0; k<K; ++k){
                    __m256d aik = _mm256_broadcast_sd(A+i*lda+k);
                    cij = _mm256_add_pd(cij, _mm256_mul_pd(_mm256_load_pd(B+k*lda+j), aik));
                }
                _mm256_store_pd(C+i*lda+j,cij);
            }
        }
        top2 += 4;
    }
    
    if((J-top2)==1){
        // printf("yu-1 ");
        for(int i=0; i<I; ++i){
            for(int k=0; k<K; ++k){
                C[i*lda+J-1] += A[i*lda+k] * B[k*lda+J-1];
            }
        }
    }else if((J-top2)==2){
        // printf("yu-2 ");
        for(int i=0; i<I; ++i){
            for(int k=0; k<K; ++k){
                C[i*lda+J-2] += A[i*lda+k] * B[k*lda+J-2];
                C[i*lda+J-1] += A[i*lda+k] * B[k*lda+J-1];
            }
        }
    }else if((J-top2)==3){
        // printf("yu-2 ");
        for(int i=0; i<I; ++i){
            for(int k=0; k<K; ++k){
                C[i*lda+J-3] += A[i*lda+k] * B[k*lda+J-3];
                C[i*lda+J-2] += A[i*lda+k] * B[k*lda+J-2];
                C[i*lda+J-1] += A[i*lda+k] * B[k*lda+J-1];
            }
        }
    }
    
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* B, double* A, double* C){

    // double* B = (double*)malloc(lda*lda*sizeof(double));

    // for(int i=0;i<lda;++i){
    //     for(int j=0;j<lda;++j){
    //         B[i*lda+j] = BB[j*lda+i];
    //     }
    // }

    for (int i = 0; i < lda; i += BSI){
        for (int j = 0; j < lda; j += BSJ){
            for (int k = 0; k < lda; k += BSK){
                int I = min (BSI, lda-i);
                int J = min (BSJ, lda-j);
                int K = min (BSK, lda-k);
                do_block(lda, I, J, K, A + i*lda + k, B + k*lda + j, C + i*lda + j);
            }
        }
    }
}
