#include "common.h"
#include <math.h>
#include <omp.h>
#include <stdio.h>
#include <mpi.h>
const char* version_name = "A naive base-line";

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
    
    ptr_t bufferx[2] = {A0, A1};
    ptr_t buffery[2] = {B0, B1};
    ptr_t bufferz[2] = {C0, C1};


    int x_start = grid_info->halo_size_x, x_end = grid_info->local_size_x + grid_info->halo_size_x;
    int y_start = grid_info->halo_size_y, y_end = grid_info->local_size_y + grid_info->halo_size_y;
    int z_start = grid_info->halo_size_z, z_end = grid_info->local_size_z + grid_info->halo_size_z;
    int ldx = grid_info->local_size_x + 2 * grid_info->halo_size_x;
    int ldy = grid_info->local_size_y + 2 * grid_info->halo_size_y;
    int ldz = grid_info->local_size_z + 2 * grid_info->halo_size_z;

    const int id = grid_info->p_id, size = grid_info->p_num;
    int cut[size + 1];
    cut[0] = 1;
    cut[grid_info->p_num] = ldx - 1;
    for(int i=1; i<size; ++i){
        cut[i] = ldz / size * i;
    }
    if(id == 0){
        printf("cut : ");
        for(int i=0; i<=size; ++i){
            printf("%d ", cut[i]);
        }
        printf("\n");
    }
    // int start = cut[id] - 1, end = cut[id + 1] + 1;
    int start = cut[id], end = cut[id + 1];
    printf("MPI - %d : [%d, %d)\n", id, start, end);

    MPI_Status status;
    int gatherv_revc[size];
    int gatherv_shift[size];
    for(int i=0; i<size; ++i){
        gatherv_revc[i] = (cut[i+1] - cut[i]) * ldx * ldy;
        gatherv_shift[i] = cut[i] * ldx * ldy;
    }

    // int num_thread_s = 16;
    // if(ldx < 300){
    //     num_thread_s = 16;
    // }else if(ldx < 400){
    //     num_thread_s = 16;
    // }else if(ldx < 600){
    //     num_thread_s = 8;
    // }else{
    //     num_thread_s = 8;
    // }

    // return bufferx[nt % 2];
    for(int t = 0; t < nt; ++t) {
        // if(id == 0){
        //     printf("MPI - %d : iter=%d\n", id, t);
        // }

        cptr_t a0 = bufferx[t % 2];
        ptr_t a1 = bufferx[(t + 1) % 2];

        cptr_t b0 = buffery[t % 2];
        ptr_t b1 = buffery[(t + 1) % 2];

        cptr_t c0 = bufferz[t % 2];
        ptr_t c1 = bufferz[(t + 1) % 2];

        #pragma omp parallel for num_threads(12)
        for(int z = start; z < end; ++z) {
            for(int y = y_start; y < y_end; ++y) {
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
        if(id != size - 1){
            MPI_Sendrecv(&a1[INDEX(0, 0, end-1, ldx, ldy)], ldx*ldy, MPI_DOUBLE, id+1, 0, &a1[INDEX(0, 0, end, ldx, ldy)], ldx*ldy, MPI_DOUBLE, id+1, 1, active_procs, &status);
            MPI_Sendrecv(&b1[INDEX(0, 0, end-1, ldx, ldy)], ldx*ldy, MPI_DOUBLE, id+1, 0, &b1[INDEX(0, 0, end, ldx, ldy)], ldx*ldy, MPI_DOUBLE, id+1, 1, active_procs, &status);
            MPI_Sendrecv(&c1[INDEX(0, 0, end-1, ldx, ldy)], ldx*ldy, MPI_DOUBLE, id+1, 0, &c1[INDEX(0, 0, end, ldx, ldy)], ldx*ldy, MPI_DOUBLE, id+1, 1, active_procs, &status);
        }
        if(id != 0){
            MPI_Sendrecv(&a1[INDEX(0, 0, start, ldx, ldy)], ldx*ldy, MPI_DOUBLE, id-1, 1, &a1[INDEX(0, 0, start-1, ldx, ldy)], ldx*ldy, MPI_DOUBLE, id-1, 0, active_procs, &status);
            MPI_Sendrecv(&b1[INDEX(0, 0, start, ldx, ldy)], ldx*ldy, MPI_DOUBLE, id-1, 1, &b1[INDEX(0, 0, start-1, ldx, ldy)], ldx*ldy, MPI_DOUBLE, id-1, 0, active_procs, &status);
            MPI_Sendrecv(&c1[INDEX(0, 0, start, ldx, ldy)], ldx*ldy, MPI_DOUBLE, id-1, 1, &c1[INDEX(0, 0, start-1, ldx, ldy)], ldx*ldy, MPI_DOUBLE, id-1, 0, active_procs, &status);
        }
        // MPI_Barrier(active_procs);
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
