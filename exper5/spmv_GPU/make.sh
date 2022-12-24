#!/bin/bash

#SBATCH -N 1
#SBATCH -n 8
#SBATCH -p gpu
#SBATCH -w gpu[03]
#SBATCH -o ./log/0-make.txt
make