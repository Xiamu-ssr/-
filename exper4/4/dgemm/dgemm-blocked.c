#include <pmmintrin.h>
#include <immintrin.h>
#include <stdio.h>
#include <string.h>
// #include <omp.h>
const char* dgemm_desc = "blocked 2 dgemm.";

#define ALIGNMENT 64
double A_cpy[1300 * 1300] __attribute__((aligned(ALIGNMENT)));
double B_cpy[1300 * 1300] __attribute__((aligned(ALIGNMENT)));
double C_cpy[1300 * 1300] __attribute__((aligned(ALIGNMENT)));

#define DOUBLE_PER_AVX512 8

#define UNROLLX 3
#define UNROLLY 8

#define BSI 192
#define BSJ 240
#define BSK 96
#define min(a,b) (((a)<(b))?(a):(b))

#define LOADC(step, c0, c1, c2)       \
  c0 = _mm512_loadu_pd(C + ((j + 0) + (i + step) * lda)); \
  c1 = _mm512_loadu_pd(C + ((j + 8) + (i + step) * lda)); \
  c2 = _mm512_loadu_pd(C + ((j + 16) + (i + step) * lda)); \

#define STOREC(step, c0, c1, c2)      \
  _mm512_storeu_pd(C + ((j + 0) + (i + step) * lda), c0); \
  _mm512_storeu_pd(C + ((j + 8) + (i + step) * lda), c1); \
  _mm512_storeu_pd(C + ((j + 16) + (i + step) * lda), c2); \

#define SET(bias)                                \
  aik0 = _mm512_set1_pd(A[k + (i + bias) * lda]); \
  aik1 = _mm512_set1_pd(A[k + (i + bias + 1) * lda]);

#define FMA(a, b, c) _mm512_fmadd_pd(a, b, c);

/* This auxiliary subroutine performs a smaller dgemm operation
 *  C := C + A * B
 * where C is M-by-N, A is M-by-K, and B is K-by-N. */
static inline void do_block(int lda, int M, int N, int K, double *A, double *B, double *C)
{
  __m512d cij[24];
  __m512d bkj0, bkj1, bkj2, aik0, aik1;
  for (int j = 0; j < N; j += 24)
  {
    for (int i = 0; i < M; i += 8)
    {
      LOADC(0, cij[0], cij[1], cij[2]);
      LOADC(1, cij[3], cij[4], cij[5]);
      LOADC(2, cij[6], cij[7], cij[8]);
      LOADC(3, cij[9], cij[10], cij[11]);
      LOADC(4, cij[12], cij[13], cij[14]);
      LOADC(5, cij[15], cij[16], cij[17]);
      LOADC(6, cij[18], cij[19], cij[20]);
      LOADC(7, cij[21], cij[22], cij[23]);
      for (int k = 0; k < K; ++k)
      {
        bkj0 = _mm512_loadu_pd(B + (j + k * lda));
        bkj1 = _mm512_loadu_pd(B + (j + 8 + k * lda));
        bkj2 = _mm512_loadu_pd(B + (j + 16 + k * lda));

        SET(0);
        cij[0] = FMA(bkj0, aik0, cij[0]);
        cij[1] = FMA(bkj1, aik0, cij[1]);
        cij[2] = FMA(bkj2, aik0, cij[2]);

        cij[3] = FMA(bkj0, aik1, cij[3]);
        cij[4] = FMA(bkj1, aik1, cij[4]);
        cij[5] = FMA(bkj2, aik1, cij[5]);

        SET(2);
        cij[6] = FMA(bkj0, aik0, cij[6]);
        cij[7] = FMA(bkj1, aik0, cij[7]);
        cij[8] = FMA(bkj2, aik0, cij[8]);

        cij[9] = FMA(bkj0, aik1, cij[9]);
        cij[10] = FMA(bkj1, aik1, cij[10]);
        cij[11] = FMA(bkj2, aik1, cij[11]);

        SET(4);
        cij[12] = FMA(bkj0, aik0, cij[12]);
        cij[13] = FMA(bkj1, aik0, cij[13]);
        cij[14] = FMA(bkj2, aik0, cij[14]);

        cij[15] = FMA(bkj0, aik1, cij[15]);
        cij[16] = FMA(bkj1, aik1, cij[16]);
        cij[17] = FMA(bkj2, aik1, cij[17]);
   
        SET(6);
        cij[18] = FMA(bkj0, aik0, cij[18]);
        cij[19] = FMA(bkj1, aik0, cij[19]);
        cij[20] = FMA(bkj2, aik0, cij[20]);

        cij[21] = FMA(bkj0, aik1, cij[21]);
        cij[22] = FMA(bkj1, aik1, cij[22]);
        cij[23] = FMA(bkj2, aik1, cij[23]);
      }
      STOREC(0, cij[0], cij[1], cij[2]);
      STOREC(1, cij[3], cij[4], cij[5]);
      STOREC(2, cij[6], cij[7], cij[8]);
      STOREC(3, cij[9], cij[10], cij[11]);
      STOREC(4, cij[12], cij[13], cij[14]);
      STOREC(5, cij[15], cij[16], cij[17]);
      STOREC(6, cij[18], cij[19], cij[20]);
      STOREC(7, cij[21], cij[22], cij[23]);
      
    }
  }
}

/* This routine performs a dgemm operation
 *  C := C + A * B
 * where A, B, and C are lda-by-lda matrices stored in column-major format. 
 * On exit, A and B maintain their input values. */  
void square_dgemm (int lda, double* B, double* A, double* C){
    const int pad = lda % 24;
    int n = lda + (pad == 0 ? 0 : 24 - pad);
    memset(C_cpy, 0, n * n * sizeof(double));

    for (int j = 0; j < n; ++j)
    {
        for (int i = 0; i < n; ++i)
        {
        if (i >= lda || j >= lda)
        {
            A_cpy[i + n * j] = 0;
            B_cpy[i + n * j] = 0;
        }
        else
        {
            A_cpy[i + n * j] = A[i + lda * j];
            B_cpy[i + n * j] = B[i + lda * j];
        }
        }
    }
// #pragma omp parallel for num_threads(4)
    for (int i = 0; i < n; i += BSI){
        for (int j = 0; j < n; j += BSJ){
            for (int k = 0; k < n; k += BSK){
                int I = min (BSI, n-i);
                int J = min (BSJ, n-j);
                int K = min (BSK, n-k);
                do_block(n, I, J, K, A_cpy + i*n + k, B_cpy + k*n + j, C_cpy + i*n + j);
            }
        }
    }

    for(int i=0; i<lda; ++i){
        for(int j=0; j<lda; ++j){
            C[i*lda+j] += C_cpy[i*n+j];
        }
    }
}
