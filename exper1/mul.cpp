#include "mul.hpp"

void Init(float* M){
    std::random_device rd;
    std::default_random_engine eng(rd());
    std::uniform_real_distribution<float> distr(FLOAT_MIN, FLOAT_MAX);
    for(int i=0; i<N*N; ++i){
        M[i] = distr(eng);
        // M[i] = .5f;
    }
}

void Mul(float* A, float* B, float* C){
    for(int i=0; i<N; ++i){
        for(int j=0; j<N; ++j){
            for(int k=0; k<N; ++k){
                C[i*N+j] += A[i*N+k] * B[k*N+j]; 
            }
        }
    }
}