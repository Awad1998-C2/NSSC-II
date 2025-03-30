#!/bin/bash

## job name

#SBATCH --job-name=benchmark_1D

## logfiles (stdout/stderr) %x=job-name %j=job-id

#SBATCH --output=stdout-%x.%j.log
#SBATCH --error=stderr-%x.%j.log

## resource requests 

#SBATCH --partition=nssc    # partition for 360.242 and 360.242
#SBATCH --nodes=1           # number of nodes
#SBATCH --ntasks=40         # number of processes
#SBATCH --cpus-per-task=1   # number of cpus per process
#SBATCH --time=00:05:00     # set time limit to 1 minute

## load modules and compilation (still on the login node)

mpic++ -std=c++17 -O3 -Wall -pedantic -march=native -ffast-math main_1D.cpp -o main_1D

resolutions=(125 250 1000 2000)
iterations=20

# Run benchmarks
for res in "${resolutions[@]}"; do
  for nprocs in 1 2 4 8 16 20 32 40; do
    echo "Running resolution=${res}, MPI processes=${nprocs}"
    mpirun -np ${nprocs} ./solverMPI 1D benchmark_1D_${res}_${nprocs} ${res} ${iterations} 0.0 1.0
  done