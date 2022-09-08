#include "./include/matrix.h"
#include <omp.h>
int main(){
    double start, end;
    struct matrix A, B, C;

    Minit(&A);
    Minit(&B);
    Minit(&C);

    start = omp_get_wtime();
    Mmul(&A, &B, &C);
    end = omp_get_wtime();

    if(Mcheck(&C)){
        printf("success\n");
        printf("time : %.4lf",end-start);
    }else{
        printf("error");
    }
    return 0;
}