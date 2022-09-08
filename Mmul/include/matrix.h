#include <stdio.h>
#include <stdlib.h>

#define D1 1023
#define D2 1023
#define D3 1023
#define D4 1023

struct matrix
{
    char *ch = (char*)malloc(sizeof(char)*D1*D1);
    int *inter1 = (int*)malloc(sizeof(int)*D2*D2);
    float *flo = (float*)malloc(sizeof(float)*D3*D3);
    int *inter2 = (int*)malloc(sizeof(int)*D4*D4);
};

//init matrix
void Minit(struct matrix* M);

// C = A * B + C
void Mmul(struct matrix* A, struct matrix* B, struct matrix* C);

//check
bool Mcheck(struct matrix* M);