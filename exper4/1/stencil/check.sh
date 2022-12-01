#!/bin/bash
#SBATCH -N 4
#SBATCH -c 32
#SBATCH -p compute
#SBATCH -w computer[07,08,09,10]
#SBATCH -o ./log/0-mpi.txt

export DAPL_DBG_TYPE=0
DATAPATH=/gpfs/home/bxjs_01/stencil_data/stencil_data

mpirun -n 4 -ppn 1 $1 7 256 256 256 16 ${DATAPATH}/stencil_data_256x256x256 ${DATAPATH}/stencil_answer_7_256x256x256_16steps
mpirun -n 4 -ppn 1 $1 7 512 512 512 16 ${DATAPATH}/stencil_data_512x512x512 ${DATAPATH}/stencil_answer_7_512x512x512_16steps