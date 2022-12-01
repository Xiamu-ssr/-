#!/bin/bash

#SBATCH -N 1
#SBATCH -n 32
#SBATCH -p compute
#SBATCH -w computer[07]
#SBATCH -o ./log/0-log.txt
./benchmark-omp 24

# low 
# advisor --collect=roofline --project-dir=./advi_resdir -- ./benchmark-blocked-5
# mid
# advisor --collect=roofline --stacks --enable-data-transfer-analysis --project-dir=./advi_resdir -- ./myApplication
# advisor --collect=map --select=has-issue --project-dir=./advi_resdir -- ./myApplication