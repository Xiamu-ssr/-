#include "../include/matrix.h"
#include <math.h>

//Minit
void Minit(struct matrix* M){
    for(int i=0; i<D1; ++i){
        for(int j=0; j<D1; ++j){
            M->ch[i*D1 + j] = 8;
        }
    }
    for(int i=0; i<D2; ++i){
        for(int j=0; j<D2; ++j){
            M->inter1[i*D2 + j] = 5;
        }
    }
    for(int i=0; i<D3; ++i){
        for(int j=0; j<D3; ++j){
            M->flo[i*D3 + j] = 2.5f;
        }
    }
    for(int i=0; i<D4; ++i){
        for(int j=0; j<D4; ++j){
            M->inter2[i*D4 + j] = 25;
        }
    }
}

// C = A * B + C
void Mmul(struct matrix* A, struct matrix* B, struct matrix* C){
    for(int i=0; i<D1; ++i){
        for(int j=0; j<D1; ++j){
            C->ch[i*D1 + j] += A->ch[i*D1 + j] * B->ch[i*D1 + j];
        }
    }
    for(int i=0; i<D2; ++i){
        for(int k=0; k<D2; ++k){
            for(int j=0; j<D2; ++j){
                C->inter1[i*D2 + j] += A->inter1[i*D2 + k] * B->inter1[k*D2 + j];
            }
        }
    }
    for(int i=0; i<D3; ++i){
        for(int k=0; k<D3; ++k){
            for(int j=0; j<D3; ++j){
                C->flo[i*D3 + j] += A->flo[i*D3 + k] * B->flo[k*D3 + j];
            }
        }
    }
    for(int i=0; i<D4; ++i){
        for(int k=0; k<D4; ++k){
            for(int j=0; j<D4; ++j){
                C->inter2[i*D4 + j] += A->inter2[i*D4 + k] * B->inter2[k*D4 + j];
            }
        }
    }
}

//check
bool Mcheck(struct matrix* M){
    for(int i=0;i<D1*D1; ++i){
        if(M->ch[i] != 72) {
            return false;
        }
    }
    for(int i=0;i<D2*D2; ++i){
        if(M->inter1[i] != 25*D2+5) {
            return false;
        }
    }
    for(int i=0;i<D3*D3; ++i){
        if(fabs(M->flo[i] - 6.25f*D3 - 2.5f) > 0.0001f) {
            return false;
        }
    }
    for(int i=0;i<D4*D4; ++i){
        if(M->inter2[i] != 625*D4+25) {
            return false;
        }
    }
    return true;
}