# Compiler & Flags
CXX = mpic++
# CXXFLAGS = -std=c++17 -O3 -march=native -ffast-math -Wall -Wextra -pedantic
CXXFLAGS = -std=c++17 -O3 -march=native -ffast-math -Wall -Wextra -pedantic -DOMPI_SKIP_MPICXX

# Directories
SRC_DIR = src
BUILD_DIR = build
DATA_DIR = data
PLOT_SCRIPT = plot.py

# Executable Name
EXEC_SEQ = $(BUILD_DIR)/solver_seq
EXEC_MPI_1D = $(BUILD_DIR)/solverMPI_1D

# Source Files
SRC_SEQ = $(SRC_DIR)/main.cpp $(SRC_DIR)/utils.cpp
SRC_MPI_1D = $(SRC_DIR)/mpi_1D.cpp $(SRC_DIR)/utils.cpp

# Object Files
OBJ_SEQ = $(BUILD_DIR)/main.o $(BUILD_DIR)/utils.o
OBJ_MPI_1D = $(BUILD_DIR)/mpi_1D.o $(BUILD_DIR)/utils.o

# Default Target
all: create_dirs $(EXEC_SEQ) $(EXEC_MPI_1D)

# Ensure necessary directories exist
create_dirs:
	@echo "Creating necessary directories..."
	mkdir -p $(BUILD_DIR)
	mkdir -p $(DATA_DIR)

# Compile object files for Sequential
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/utils.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Compile object files for MPI 1D
$(BUILD_DIR)/mpi_1D.o: $(SRC_DIR)/mpi_1D.cpp $(SRC_DIR)/utils.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Shared utils.o
$(BUILD_DIR)/utils.o: $(SRC_DIR)/utils.cpp $(SRC_DIR)/utils.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link Sequential executable
$(EXEC_SEQ): $(OBJ_SEQ)
	$(CXX) $(CXXFLAGS) $(OBJ_SEQ) -o $(EXEC_SEQ)

# Link MPI executable
$(EXEC_MPI_1D): $(OBJ_MPI_1D)
	$(CXX) $(CXXFLAGS) $(OBJ_MPI_1D) -o $(EXEC_MPI_1D)

# Run Sequential solver with default parameters
run_seq: $(EXEC_SEQ)
	@echo "Running sequential solver..."
	$(EXEC_SEQ) 1D benchmark1 250 100 0.0 1.0

# Run MPI 1D solver with default parameters (e.g., 4 processes)
run_mpi_1d: $(EXEC_MPI_1D)
	@echo "Running MPI 1D solver with 4 processes..."
	mpirun -np 4 $(EXEC_MPI_1D) 1D benchmarkMPI1D 125 10000 0.0 1.0

# Generate plots from MPI 1D results
plot_mpi:
	@echo "Generating MPI plots..."
	python3 $(PLOT_SCRIPT) -i data/benchmarkMPI1D.csv -o data/benchmarkMPI1D.png

# Clean build and data directories
clean:
	@echo "Cleaning build and data directories..."
	rm -rf $(BUILD_DIR) $(DATA_DIR)

.PHONY: all create_dirs run_seq run_mpi_1d plot_mpi clean
