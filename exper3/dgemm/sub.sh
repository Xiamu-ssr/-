#!/bin/bash

#SBATCH -n 1
#SBATCH -N 1
#SBATCH -p compute
#SBATCH -w computer[16]
#SBATCH -o ./log.txt
./benchmark-blocked

# low 
# advisor --collect=roofline --project-dir=./advi_resdir -- ./benchmark-blocked-5
# mid
# advisor --collect=roofline --stacks --enable-data-transfer-analysis --project-dir=./advi_resdir -- ./myApplication
# advisor --collect=map --select=has-issue --project-dir=./advi_resdir -- ./myApplication