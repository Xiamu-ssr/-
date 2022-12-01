#include "common.h"
#include <math.h>
#include <immintrin.h>
#include <stdio.h>
#include <omp.h>
#include <stdlib.h>
const char* version_name = "A naive base-line";

#define LOADA(t1, t2, t3, t4, t5, t6, t7) \
    A_1 = _mm512_set1_pd(t1);\
    A_2 = _mm512_set1_pd(t2);\
    A_3 = _mm512_set1_pd(t3);\
    A_4 = _mm512_set1_pd(t4);\
    A_5 = _mm512_set1_pd(t5);\
    A_6 = _mm512_set1_pd(t6);\
    A_7 = _mm512_set1_pd(t7);

#define LOADXY(source) \
    xy1 = _mm512_loadu_pd(source+INDEX(x, y, z, ldx, ldy));\
    xy2 = _mm512_loadu_pd(source+INDEX((x-1), y, z, ldx, ldy));\
    xy3 = _mm512_loadu_pd(source+INDEX((x+1), y, z, ldx, ldy));\
    xy4 = _mm512_loadu_pd(source+INDEX(x, (y-1), z, ldx, ldy));\
    xy5 = _mm512_loadu_pd(source+INDEX(x, (y+1), z, ldx, ldy));\
    xy6 = _mm512_loadu_pd(source+INDEX(x, y, (z-1), ldx, ldy));\
    xy7 = _mm512_loadu_pd(source+INDEX(x, y, (z+1), ldx, ldy));

#define FMAXY(tag) \
    tag = _mm512_mul_pd(xy1, A_1);\
    tag = _mm512_fmadd_pd(xy2, A_2, tag);\
    tag = _mm512_fmadd_pd(xy3, A_3, tag);\
    tag = _mm512_fmadd_pd(xy4, A_4, tag);\
    tag = _mm512_fmadd_pd(xy5, A_5, tag);\
    tag = _mm512_fmadd_pd(xy6, A_6, tag);\
    tag = _mm512_fmadd_pd(xy7, A_7, tag);

void check_simd(__m512d s, char* name){
    data_t tmp[Dper512];
    _mm512_storeu_pd(tmp, s);
    printf("%s : ",name);
    for(int i=0; i<Dper512; ++i){
        printf("%.2lf ",tmp[i]);
    }
    printf("\n");
}

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    /* Naive implementation uses Process 0 to do all computations */
    if(grid_info->p_id == 0) {
        grid_info->local_size_x = grid_info->global_size_x;
        grid_info->local_size_y = grid_info->global_size_y;
        grid_info->local_size_z = grid_info->global_size_z;
    } else {
        grid_info->local_size_x = 0;
        grid_info->local_size_y = 0;
        grid_info->local_size_z = 0;
    }
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = 0;
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

ptr_t stencil_7(ptr_t A0, ptr_t A1,ptr_t B0, ptr_t B1,ptr_t C0, ptr_t C1, const dist_grid_info_t *grid_info, int nt) {
    ptr_t bufferx[2] = {A0, A1};
    ptr_t buffery[2] = {B0, B1};
    ptr_t bufferz[2] = {C0, C1};

    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    printf("iters=%d end=%d - start=%d ldx=%d - ldy=%d ldz=%d\n",nt, x_end,x_start, ldx, ldy, ldz);

    for(int t = 0; t < nt; ++t) {
        cptr_t a0 = bufferx[t % 2];
        ptr_t a1 = bufferx[(t + 1) % 2];

        cptr_t b0 = buffery[t % 2];
        ptr_t b1 = buffery[(t + 1) % 2];

        cptr_t c0 = bufferz[t % 2];
        ptr_t c1 = bufferz[(t + 1) % 2];
        for(int z = z_start; z < z_end; ++z) {
            for(int y = y_start; y < y_end; ++y) {
                for(int x = x_start; x < x_end; x+=Dper512) {
                    data_t a77[Dper512], b77[Dper512], c77[Dper512];
                    __m512d tmp, xy1, xy2, xy3, xy4, xy5, xy6, xy7;
                    __m512d A_1, A_2, A_3, A_4, A_5, A_6, A_7;

                    //a
                    LOADA(ALPHA_ZZZ, ALPHA_NZZ, ALPHA_PZZ, ALPHA_ZNZ, ALPHA_ZPZ, ALPHA_ZZN, ALPHA_ZZP);
                    LOADXY(a0);
                    tmp = _mm512_set1_pd(0);
                    FMAXY(tmp);
                    _mm512_storeu_pd(a77, tmp);
                    //b
                    LOADA(ALPHA_PNZ, ALPHA_NPZ, ALPHA_PPZ, ALPHA_NZN, ALPHA_PZN, ALPHA_PZP, ALPHA_NZP);
                    LOADXY(b0);
                    tmp = _mm512_set1_pd(0);
                    FMAXY(tmp);
                    _mm512_storeu_pd(b77, tmp);
                    //c
                    LOADA(ALPHA_PNN, ALPHA_PPN, ALPHA_PPN, ALPHA_NNP, ALPHA_PNP, ALPHA_NPP, ALPHA_PPP);
                    LOADXY(c0);
                    tmp = _mm512_set1_pd(0);
                    FMAXY(tmp);
                    _mm512_storeu_pd(c77, tmp);

                    for(int i=0; i<Dper512; ++i){
                        a1[INDEX(x+i, y, z, ldx, ldy)] = a77[i]  +  (b77[i] * c77[i]) / (b77[i] + c77[i]); //sqrt
                        b1[INDEX(x+i, y, z, ldx, ldy)] = b77[i]  +  (a77[i] * c77[i]) / (a77[i] + c77[i]); //sqrt
                        c1[INDEX(x+i, y, z, ldx, ldy)] = c77[i]  +  (a77[i] * b77[i]) / (a77[i] + b77[i]); //sqrt
                    }
                }
            }
        }
    }
    return bufferx[nt%2];
}