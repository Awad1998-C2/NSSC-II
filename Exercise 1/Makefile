# Compiler & Flags
CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -ffast-math -Wall -Wextra -pedantic

# Directories
SRC_DIR = src
BUILD_DIR = build
DATA_DIR = data
PLOT_SCRIPT = plot.py

# Executable Name
EXEC = $(BUILD_DIR)/solver

# Source Files
SRC = $(SRC_DIR)/main.cpp $(SRC_DIR)/utils.cpp

# Object Files
OBJ = $(BUILD_DIR)/main.o $(BUILD_DIR)/utils.o

# Default Target
all: create_dirs $(EXEC)

# Ensure necessary directories exist
create_dirs:
	@echo "Creating necessary directories..."
	mkdir -p $(BUILD_DIR)
	mkdir -p $(DATA_DIR)

# Compile object files
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.cpp $(SRC_DIR)/utils.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(BUILD_DIR)/utils.o: $(SRC_DIR)/utils.cpp $(SRC_DIR)/utils.hpp
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Link the executable
$(EXEC): $(OBJ)
	$(CXX) $(CXXFLAGS) $(OBJ) -o $(EXEC)

# Run the program with default parameters
run: $(EXEC)
	@echo "Running solver with default parameters..."
	$(EXEC) 1D benchmark1 128 200000 0.0 1.0

# Generate plots
plot:
	@echo "Generating plots..."
	python3 $(PLOT_SCRIPT) -i data/benchmark1.csv -o data/benchmark1.png

# Clean build and data directories
clean:
	@echo "Cleaning build and data directories..."
	rm -rf $(BUILD_DIR) $(DATA_DIR)

.PHONY: all create_dirs run plot clean
