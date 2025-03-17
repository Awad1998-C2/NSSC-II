CXX = g++
CXXFLAGS = -std=c++17 -O3 -march=native -ffast-math -Wall -Wextra -pedantic

SRC_DIR = src
BUILD_DIR = build
DATA_DIR = data
PLOT_SCRIPT = plot.py

EXEC = $(BUILD_DIR)/solverMPI
SRC = $(SRC_DIR)/main.cpp

all: create_dirs $(EXEC)

create_dirs:
	@echo "Ensuring directories exist: $(BUILD_DIR) and $(DATA_DIR)"
	mkdir -p $(BUILD_DIR)
	mkdir -p $(DATA_DIR)

$(EXEC): $(SRC)
	@echo "Compiling solver..."
	$(CXX) $(CXXFLAGS) $^ -o $@ $(LDFLAGS)

run: $(EXEC)
	@echo "Running solver with default parameters..."
	$(EXEC) 1D benchmark1 128 200000 0.0 1.0

plot:
	@echo "Generating plots..."
	python3 $(PLOT_SCRIPT) -i data/benchmark1.csv -o data/benchmark1.png
	
clean:
	@echo "Cleaning build and data directories..."
	rm -rf $(BUILD_DIR) $(DATA_DIR)

.PHONY: all create_dirs run plot clean
