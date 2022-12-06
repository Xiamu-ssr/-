#!/bin/bash
#SBATCH -N 2
#SBATCH -c 32
#SBATCH -p compute
#SBATCH -w computer[15,16]

export DAPL_DBG_TYPE=0
export OMP_PLACES=cores
DATAPATH=/gpfs/home/bxjs_01/stencil_data/stencil_data

mpirun -n 4 -ppn 2 $1 7 256 256 256 100 ${DATAPATH}/stencil_data_256x256x256
mpirun -n 4 -ppn 2 $1 7 384 384 384 100 ${DATAPATH}/stencil_data_384x384x384
mpirun -n 4 -ppn 2 $1 7 512 512 512 100 ${DATAPATH}/stencil_data_512x512x512
mpirun -n 4 -ppn 2 $1 7 768 768 768 100 ${DATAPATH}/stencil_data_768x768x768
