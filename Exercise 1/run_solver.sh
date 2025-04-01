#!/bin/bash

# Usage: ./run_solver.sh <np> <mode> <file> <N> <max_iter> <a> <b>
# Example: ./run_solver.sh 4 2D test.txt 100 20 0.0 1.0

np="$1"
mode="$2"
file="$3"
N="$4"
max_iter="$5"
a="$6"
b="$7"

# Check if np is a number
if ! [[ "$np" =~ ^[0-9]+$ ]]; then
  echo "First argument must be the number of processes"
  exit 1
fi

# Function to check if np can form a 2D grid
is_2d_grid() {
  for ((i=2; i<=np/2; i++)); do
    if (( np % i == 0 )); then
      return 0
    fi
  done
  return 1
}

# Dispatch logic
if (( np == 1 )); then
  echo "[INFO] Running sequential solver (main)"
  ./main "$mode" "$file" "$N" "$max_iter" "$a" "$b"

elif (( np >= 2 && np <= 40 )); then
  if [[ "$mode" == "2D" ]]; then
    if is_2d_grid; then
      echo "[INFO] Running 2D MPI solver (2D)"
      mpirun -np "$np" ./2D "$mode" "$file" "$N" "$max_iter" "$a" "$b"
    else
      echo "[INFO] Cannot create 2D grid with $np processes. Falling back to 1D solver (main_1D)"
      mpirun -np "$np" ./main_1D "$mode" "$file" "$N" "$max_iter" "$a" "$b"
    fi
  else
    echo "[INFO] Running 1D MPI solver (main_1D)"
    mpirun -np "$np" ./main_1D "$mode" "$file" "$N" "$max_iter" "$a" "$b"
  fi
else
  echo "[ERROR] Number of processes must be between 1 and 40"
  exit 1
fi
