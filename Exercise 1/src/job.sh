#!/bin/bash

## job name
#SBATCH --job-name=benchmark_1D

## logfiles (stdout/stderr) %x=job-name %j=job-id
#SBATCH --output=stdout-%x.%j.log
#SBATCH --error=stderr-%x.%j.log

## resource requests 
#SBATCH --partition=nssc         # partition for the cluster
#SBATCH --nodes=1                # number of nodes
#SBATCH --ntasks=40              # number of processes
#SBATCH --cpus-per-task=1        # number of cpus per process
#SBATCH --time=00:10:00          # set time limit

## optional: load relevant modules, e.g.:
# module load mpi/mpich-x86_64 pmi/pmix-x86_64

## Compilation
mpic++ -std=c++17 -O3 -Wall -pedantic -march=native -ffast-math main_1D.cpp -o main_1D

# resolutions=(125 250 1000 2000)
resolutions=(2000)

iterations=100

## Run benchmarks
for res in "${resolutions[@]}"; do
  for ((nprocs = 1; nprocs <= 40; nprocs++)); do
    echo "Running resolution=${res}, MPI processes=${nprocs}"
    mpirun -np ${nprocs} ./main_1D 1D benchmark_1D_${res}_${nprocs} ${res} ${iterations} 0.0 1.0
  done
done
