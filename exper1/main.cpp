#include "mul.hpp"
#include <time.h>

int main(){

    float* A = (float*)malloc(N * N * sizeof(float));
    float* B = (float*)malloc(N * N * sizeof(float));
    float* C = (float*)malloc(N * N * sizeof(float));

    Init(A);
    Init(B);
    Init(C);

    Mul(A, B, C);
    // for(int i=0;i<N*N;++i){
    //     printf("A[%d] = %.5lf\n",i,A[i]);
    // }
    // for(int i=0;i<N*N;++i){
    //     printf("B[%d] = %.5lf\n",i,B[i]);
    // }
    // for(int i=0;i<10;++i){
    //     printf("C[%d] = %.5lf\n",i,C[i]);
    // }
    
    return 0;
}