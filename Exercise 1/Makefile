# Compiler
CXX = mpic++
CXXFLAGS = -std=c++17 -O3 -Wall -pedantic -march=native -ffast-math

# Source directories
SRC_DIR = src
TASK3_DIR = task3

# Targets and their source files
MAIN = main
MAIN_1D = main_1D
2D = 2D

.PHONY: all clean

all: $(MAIN) $(MAIN_1D) $(2D)

$(MAIN): $(SRC_DIR)/main.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(MAIN_1D): $(SRC_DIR)/main_1D.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

$(2D): $(TASK3_DIR)/2D.cpp
	$(CXX) $(CXXFLAGS) -o $@ $<

clean:
	rm -f $(MAIN) $(MAIN_1D) $(2D)
