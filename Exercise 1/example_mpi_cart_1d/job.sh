#!/bin/bash
#SBATCH --job-name=example_mpi_cart_1d
#SBATCH --output=%x-stdout-%j.log
#SBATCH  --error=%x-stderr-%j.log
#SBATCH --nodelist=tcad31
#SBATCH --ntasks=4
#SBATCH --cpus-per-task=1
#SBATCH --time=00:02:00 

module load mpi/mpich-x86_64 pmi/pmix-x86_64

mpic++ -std=c++17 main.cpp -o main

mpirun hostname 
mpirun ./main 

