#include "common.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <mpi.h>
#include <string.h>
#include <algorithm>
const char* version_name = "A naive base-line";

#ifndef BT
    #define BT 5
#endif
#ifndef BX
    #define BX 24
#endif
#ifndef BY
    #define BY 16
#endif
#ifndef BZ
    #define BZ 16
#endif

#define BUF_X (BX+2*BT)
#define BUF_Y (BY+2*BT)
#define BUF_Z (BZ+2*BT)
#define BUF_SIZE (BUF_X*BUF_Y*BUF_Z)
// #define MAX(a,b) ((a)<(b)) ? (b) : (a)
// #define MIN(a,b) ((a)>(b)) ? (b) : (a)

// data_t buf_A0[BUF_SIZE] = {}, buf_B0[BUF_SIZE] = {}, buf_C0[BUF_SIZE] = {};
// data_t buf_A1[BUF_SIZE] = {}, buf_B1[BUF_SIZE] = {}, buf_C1[BUF_SIZE] = {};

void create_dist_grid(dist_grid_info_t *grid_info, int stencil_type) {
    /* Naive implementation uses Process 0 to do all computations */
    // if(grid_info->p_id == 0) {
        grid_info->local_size_x = grid_info->global_size_x;
        grid_info->local_size_y = grid_info->global_size_y;
        grid_info->local_size_z = grid_info->global_size_z;
    // } else {
    //     grid_info->local_size_x = 0;
    //     grid_info->local_size_y = 0;
    //     grid_info->local_size_z = 0;
    // }
    grid_info->offset_x = 0;
    grid_info->offset_y = 0;
    grid_info->offset_z = 0;
    grid_info->halo_size_x = 1;
    grid_info->halo_size_y = 1;
    grid_info->halo_size_z = 1;
}

void destroy_dist_grid(dist_grid_info_t *grid_info) {

}

ptr_t stencil_7(ptr_t A0, ptr_t A1,ptr_t B0, ptr_t B1,ptr_t C0, ptr_t C1, const dist_grid_info_t *grid_info, int nt, MPI_Comm active_procs) {
    // int BT = bt, BX = bx, BY = by, BZ = bz;
    // int BUF_X = BX+2*BT, BUF_Y = BY+2*BY, BUF_Z = BZ+2*BZ;
    // int BUF_SIZE = BUF_X*BUF_Y*BUF_Z;
    ptr_t bufferx[2] = {A0, A1};
    ptr_t buffery[2] = {B0, B1};
    ptr_t bufferz[2] = {C0, C1};


    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    //calculate MPI info for each PROC
    const int id = grid_info->p_id, size = grid_info->p_num;
    int cut[size + 1];
    cut[0] = 1;
    cut[grid_info->p_num] = ldx - 1;
    for(int i=1; i<size; ++i){
        cut[i] = ldz / size * i;
    }
    if(id == 0){
        printf("\t--INFO--\n");
        printf("\tBX = %d\n\tBY = %d\n\tBZ = %d\n\tBT = %d\n", BX, BY, BZ, BT);
        printf("\tnt = %d\n",nt);
        printf("\tx : [%d - %d)\n", x_start, x_end);
        printf("\ty : [%d - %d)\n", y_start, y_end);
        printf("\tz : [%d - %d)\n", z_start, z_end);
        printf("\tcut : ");
        for(int i=0; i<=size; ++i){
            printf("%d ", cut[i]);
        }
        printf("\n");
    }
    //[start, end) index of this PRO of MPI should calculate on Z axis
    // int start = cut[id] - 1, end = cut[id + 1] + 1;
    int start = cut[id], end = cut[id + 1];
    printf("\tMPI - %d : [%d, %d)\n", id, start, end);

    MPI_Status status;
    int gatherv_revc[size];
    int gatherv_shift[size];
    for(int i=0; i<size; ++i){
        gatherv_revc[i] = (cut[i+1] - cut[i]) * ldx * ldy;
        gatherv_shift[i] = cut[i] * ldx * ldy;
    }

    // calculate %
    for(int t = 0; t < nt%BT; ++t) {

        cptr_t a0 = bufferx[t % 2];
        ptr_t a1 = bufferx[(t + 1) % 2];

        cptr_t b0 = buffery[t % 2];
        ptr_t b1 = buffery[(t + 1) % 2];

        cptr_t c0 = bufferz[t % 2];
        ptr_t c1 = bufferz[(t + 1) % 2];
        
        for(int z = z_start; z < z_end; ++z) {
            for(int y = y_start; y < y_end; ++y) {
                #pragma omp simd
                for(int x = x_start; x < x_end; ++x) {

                    data_t a7,b7,c7;

                    a7 \
                        = ALPHA_ZZZ * a0[INDEX(x, y, z, ldx, ldy)] \
                        + ALPHA_NZZ * a0[INDEX(x-1, y, z, ldx, ldy)] \
                        + ALPHA_PZZ * a0[INDEX(x+1, y, z, ldx, ldy)] \
                        + ALPHA_ZNZ * a0[INDEX(x, y-1, z, ldx, ldy)] \
                        + ALPHA_ZPZ * a0[INDEX(x, y+1, z, ldx, ldy)] \
                        + ALPHA_ZZN * a0[INDEX(x, y, z-1, ldx, ldy)] \
                        + ALPHA_ZZP * a0[INDEX(x, y, z+1, ldx, ldy)];

                     b7 \
                        = ALPHA_PNZ * b0[INDEX(x, y, z, ldx, ldy)] \
                        + ALPHA_NPZ * b0[INDEX(x-1, y, z, ldx, ldy)] \
                        + ALPHA_PPZ * b0[INDEX(x+1, y, z, ldx, ldy)] \
                        + ALPHA_NZN * b0[INDEX(x, y-1, z, ldx, ldy)] \
                        + ALPHA_PZN * b0[INDEX(x, y+1, z, ldx, ldy)] \
                        + ALPHA_PZP * b0[INDEX(x, y, z-1, ldx, ldy)] \
                        + ALPHA_NZP * b0[INDEX(x, y, z+1, ldx, ldy)];

                    c7 \
                        = ALPHA_PNN * c0[INDEX(x, y, z, ldx, ldy)] \
                        + ALPHA_PPN * c0[INDEX(x-1, y, z, ldx, ldy)] \
                        + ALPHA_PPN * c0[INDEX(x+1, y, z, ldx, ldy)] \
                        + ALPHA_NNP * c0[INDEX(x, y-1, z, ldx, ldy)] \
                        + ALPHA_PNP * c0[INDEX(x, y+1, z, ldx, ldy)] \
                        + ALPHA_NPP * c0[INDEX(x, y, z-1, ldx, ldy)] \
                        + ALPHA_PPP * c0[INDEX(x, y, z+1, ldx, ldy)];

                    a1[INDEX(x, y, z, ldx, ldy)] = a7  +  (b7 * c7) / (b7 + c7); //sqrt
                    b1[INDEX(x, y, z, ldx, ldy)] = b7  +  (a7 * c7) / (a7 + c7); //sqrt
                    c1[INDEX(x, y, z, ldx, ldy)] = c7  +  (a7 * b7) / (a7 + b7); //sqrt

                }
            }
        }
    }

    //calculate all
    for(int t = nt%BT; t < nt; t+=BT) {
        ptr_t a0 = bufferx[t % 2];
        ptr_t a1 = bufferx[(t+1) % 2];

        ptr_t b0 = buffery[t % 2];
        ptr_t b1 = buffery[(t+1) % 2];

        ptr_t c0 = bufferz[t % 2];
        ptr_t c1 = bufferz[(t+1) % 2];
        if(id == 0){
            printf("MPI - %d : calculate T = %d\n", id, t);
        }
        #pragma omp parallel for num_threads(16) collapse(2) schedule(static)
        for(int z = start; z < end; z+=BZ) {
            for(int y = y_start; y < y_end; y+=BY) {
                for(int x = x_start; x < x_end; x+=BX) {
                    
                    //right
                    //[buf_start, buf_end] index in data of start and end which should copy from data to buf
                    int buf_x_start = x-BT, buf_x_end = x + BX-1  + BT;
                    int buf_y_start = y-BT, buf_y_end = y + BY-1  + BT;
                    int buf_z_start = z-BT, buf_z_end = z + BZ-1  + BT;

                    //right
                    //[data_start, data_end] index in data of start and end which really copy from data to buf
                    int data_x_head = std::max(0, buf_x_start), data_x_tail = std::min(x_end, buf_x_end);
                    int data_y_head = std::max(0, buf_y_start), data_y_tail = std::min(y_end, buf_y_end);
                    int data_z_head = std::max(0, buf_z_start), data_z_tail = std::min(z_end, buf_z_end);

                    //[buf_head, buf_tail] index in buf of start and end which really buf in BUF
                    int buf_x_head = data_x_head - buf_x_start, buf_x_tail = BUF_X-1 - (buf_x_end - data_x_tail), buf_x_len = buf_x_tail - buf_x_head + 1;
                    int buf_y_head = data_y_head - buf_y_start, buf_y_tail = BUF_Y-1 - (buf_y_end - data_y_tail), buf_y_len = buf_y_tail - buf_y_head + 1;
                    int buf_z_head = data_z_head - buf_z_start, buf_z_tail = BUF_Z-1 - (buf_z_end - data_z_tail), buf_z_len = buf_z_tail - buf_z_head + 1;
                    // printf("[%d, %d] [%d, %d] [%d, %d] \n", buf_x_start, buf_x_end, buf_y_start, buf_y_end, buf_z_start, buf_z_end);
                    // printf("[%d, %d] [%d, %d] [%d, %d] \n", data_x_head, data_x_tail, data_y_head, data_y_tail, data_z_head, data_z_tail);
                    // printf("[%d, %d] [%d, %d] [%d, %d] \n", buf_x_head, buf_x_tail, buf_y_head, buf_y_tail, buf_z_head, buf_z_tail);

                    data_t buf_A0[BUF_SIZE] = {}, buf_B0[BUF_SIZE] = {}, buf_C0[BUF_SIZE] = {};
                    data_t buf_A1[BUF_SIZE] = {}, buf_B1[BUF_SIZE] = {}, buf_C1[BUF_SIZE] = {};
                    // memset(buf_A0, 0, BUF_SIZE*sizeof(data_t));
                    // memset(buf_B0, 0, BUF_SIZE*sizeof(data_t));
                    // memset(buf_C0, 0, BUF_SIZE*sizeof(data_t));
                    // memset(buf_A1, 0, BUF_SIZE*sizeof(data_t));
                    // memset(buf_B1, 0, BUF_SIZE*sizeof(data_t));
                    // memset(buf_C1, 0, BUF_SIZE*sizeof(data_t));
                    for(int b_z=buf_z_head; b_z<=buf_z_tail; ++b_z){
                        for(int b_y=buf_y_head; b_y<=buf_y_tail; ++b_y){
                            memcpy(buf_A0+INDEX(buf_x_head, b_y, b_z, BUF_X, BUF_Y), a0+INDEX(data_x_head, buf_y_start+b_y, buf_z_start+b_z, ldx, ldy), buf_x_len*sizeof(data_t));
                            memcpy(buf_B0+INDEX(buf_x_head, b_y, b_z, BUF_X, BUF_Y), b0+INDEX(data_x_head, buf_y_start+b_y, buf_z_start+b_z, ldx, ldy), buf_x_len*sizeof(data_t));
                            memcpy(buf_C0+INDEX(buf_x_head, b_y, b_z, BUF_X, BUF_Y), c0+INDEX(data_x_head, buf_y_start+b_y, buf_z_start+b_z, ldx, ldy), buf_x_len*sizeof(data_t));
                        }
                    }
                    ptr_t buf_a0 = buf_A0, buf_b0 = buf_B0, buf_c0 = buf_C0;
                    ptr_t buf_a1 = buf_A1, buf_b1 = buf_B1, buf_c1 = buf_C1;
                    
                    //head and tail is right
                    #pragma unroll(BT)
                    for(int tt=0; tt<BT; ++tt){
                        // int xx_start = 1+tt, xx_end = BUF_X-2-tt;
                        // int yy_start = 1+tt, yy_end = BUF_Y-2-tt;
                        // int zz_start = 1+tt, zz_end = BUF_Z-2-tt;
                        int xx_start = std::max(1+tt, buf_x_head+1), xx_end = std::min(BUF_X-2-tt, buf_x_tail-1);
                        int yy_start = std::max(1+tt, buf_y_head+1), yy_end = std::min(BUF_Y-2-tt, buf_y_tail-1);
                        int zz_start = std::max(1+tt, buf_z_head+1), zz_end = std::min(BUF_Z-2-tt, buf_z_tail-1);
                        for(int zz=zz_start; zz<=zz_end; ++zz){
                            for(int yy=yy_start; yy<=yy_end; ++yy){
                                #pragma omp simd
                                for(int xx=xx_start; xx<=xx_end; ++xx){
                                    //calculate is right
                                    data_t a7,b7,c7;
                                    a7 \
                                        = ALPHA_ZZZ * buf_a0[INDEX(xx, yy, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_NZZ * buf_a0[INDEX(xx-1, yy, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_PZZ * buf_a0[INDEX(xx+1, yy, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_ZNZ * buf_a0[INDEX(xx, yy-1, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_ZPZ * buf_a0[INDEX(xx, yy+1, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_ZZN * buf_a0[INDEX(xx, yy, zz-1, BUF_X, BUF_Y)] \
                                        + ALPHA_ZZP * buf_a0[INDEX(xx, yy, zz+1, BUF_X, BUF_Y)];

                                    b7 \
                                        = ALPHA_PNZ * buf_b0[INDEX(xx, yy, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_NPZ * buf_b0[INDEX(xx-1, yy, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_PPZ * buf_b0[INDEX(xx+1, yy, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_NZN * buf_b0[INDEX(xx, yy-1, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_PZN * buf_b0[INDEX(xx, yy+1, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_PZP * buf_b0[INDEX(xx, yy, zz-1, BUF_X, BUF_Y)] \
                                        + ALPHA_NZP * buf_b0[INDEX(xx, yy, zz+1, BUF_X, BUF_Y)];

                                    c7 \
                                        = ALPHA_PNN * buf_c0[INDEX(xx, yy, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_PPN * buf_c0[INDEX(xx-1, yy, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_PPN * buf_c0[INDEX(xx+1, yy, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_NNP * buf_c0[INDEX(xx, yy-1, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_PNP * buf_c0[INDEX(xx, yy+1, zz, BUF_X, BUF_Y)] \
                                        + ALPHA_NPP * buf_c0[INDEX(xx, yy, zz-1, BUF_X, BUF_Y)] \
                                        + ALPHA_PPP * buf_c0[INDEX(xx, yy, zz+1, BUF_X, BUF_Y)];
                                    
                                    buf_a1[INDEX(xx, yy, zz, BUF_X, BUF_Y)] = a7  +  (b7 * c7) / (b7 + c7); //sqrt
                                    buf_b1[INDEX(xx, yy, zz, BUF_X, BUF_Y)] = b7  +  (a7 * c7) / (a7 + c7); //sqrt
                                    buf_c1[INDEX(xx, yy, zz, BUF_X, BUF_Y)] = c7  +  (a7 * b7) / (a7 + b7); //sqrt
                                }
                            }
                        }
                        std::swap(buf_a0, buf_a1);
                        std::swap(buf_b0, buf_b1);
                        std::swap(buf_c0, buf_c1);
                    }
                    // int b_z_start = BT, b_z_end = BT+BZ-1;
                    // int b_y_start = BT, b_y_end = BT+BY-1;
                    //[b_start, b_end] index in buf1 of start and end which really copy back
                    int b_x_start = std::max(BT, buf_x_head+1), b_x_end = std::min(BT+BX-1, buf_x_tail-1), b_x_len = b_x_end - b_x_start + 1;
                    int b_y_start = std::max(BT, buf_y_head+1), b_y_end = std::min(BT+BY-1, buf_y_tail-1);
                    int b_z_start = std::max(BT, buf_z_head+1), b_z_end = std::min(BT+BZ-1, buf_z_tail-1);
                    for(int b_z=b_z_start; b_z<=b_z_end; ++b_z){
                        for(int b_y=b_y_start; b_y<=b_y_end; ++b_y){
                            memcpy(a1+INDEX(buf_x_start+BT, buf_y_start+b_y, buf_z_start+b_z, ldx, ldy), buf_a0+INDEX(BT, b_y, b_z, BUF_X, BUF_Y), b_x_len*sizeof(data_t));
                            memcpy(b1+INDEX(buf_x_start+BT, buf_y_start+b_y, buf_z_start+b_z, ldx, ldy), buf_b0+INDEX(BT, b_y, b_z, BUF_X, BUF_Y), b_x_len*sizeof(data_t));
                            memcpy(c1+INDEX(buf_x_start+BT, buf_y_start+b_y, buf_z_start+b_z, ldx, ldy), buf_c0+INDEX(BT, b_y, b_z, BUF_X, BUF_Y), b_x_len*sizeof(data_t));
                        }
                    }
                }
            }
        }
        //bottom
        if(id != size - 1){
            MPI_Sendrecv(&a1[INDEX(0, 0, end-BT, ldx, ldy)], ldx*ldy*BT, MPI_DOUBLE, id+1, 0, &a1[INDEX(0, 0, end, ldx, ldy)], ldx*ldy*BT, MPI_DOUBLE, id+1, 1, active_procs, &status);
            MPI_Sendrecv(&b1[INDEX(0, 0, end-BT, ldx, ldy)], ldx*ldy*BT, MPI_DOUBLE, id+1, 0, &b1[INDEX(0, 0, end, ldx, ldy)], ldx*ldy*BT, MPI_DOUBLE, id+1, 1, active_procs, &status);
            MPI_Sendrecv(&c1[INDEX(0, 0, end-BT, ldx, ldy)], ldx*ldy*BT, MPI_DOUBLE, id+1, 0, &c1[INDEX(0, 0, end, ldx, ldy)], ldx*ldy*BT, MPI_DOUBLE, id+1, 1, active_procs, &status);
        }
        //top
        if(id != 0){
            MPI_Sendrecv(&a1[INDEX(0, 0, start, ldx, ldy)], ldx*ldy*BT, MPI_DOUBLE, id-1, 1, &a1[INDEX(0, 0, start-BT, ldx, ldy)], ldx*ldy*BT, MPI_DOUBLE, id-1, 0, active_procs, &status);
            MPI_Sendrecv(&b1[INDEX(0, 0, start, ldx, ldy)], ldx*ldy*BT, MPI_DOUBLE, id-1, 1, &b1[INDEX(0, 0, start-BT, ldx, ldy)], ldx*ldy*BT, MPI_DOUBLE, id-1, 0, active_procs, &status);
            MPI_Sendrecv(&c1[INDEX(0, 0, start, ldx, ldy)], ldx*ldy*BT, MPI_DOUBLE, id-1, 1, &c1[INDEX(0, 0, start-BT, ldx, ldy)], ldx*ldy*BT, MPI_DOUBLE, id-1, 0, active_procs, &status);
        }
    }
    printf("MPI - %d : OVER\n", id);
    MPI_Barrier(active_procs);
    if(id == 0){
        MPI_Gatherv(MPI_IN_PLACE, (end-start)*ldx*ldy, MPI_DOUBLE, &bufferx[nt%2][INDEX(0, 0, 0, ldx, ldy)], gatherv_revc, gatherv_shift, MPI_DOUBLE, 0, active_procs);
    }else{
        MPI_Gatherv(&bufferx[nt%2][INDEX(0, 0, start, ldx, ldy)], (end-start)*ldx*ldy, MPI_DOUBLE, NULL, gatherv_revc, gatherv_shift, MPI_DOUBLE, 0, active_procs);
    }
    MPI_Barrier(active_procs); 
    if(id == 0){
        printf("MPI - %d : ALLOVER\n", id);
    }

    return bufferx[nt % 2];
}
