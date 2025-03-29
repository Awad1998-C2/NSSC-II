#ifndef UTILS_HPP
#define UTILS_HPP

#include <vector>
#include <string>

// Initialize the grid with boundary conditions
std::vector<double> init(size_t N, double W, double E);

// Perform Jacobi iteration
void jacobi_iter_1_process(const std::vector<double>& xold, std::vector<double>& xnew, size_t N, bool residual = false);

// Write results to CSV
void write(const std::vector<double>& x, size_t N, const std::string& name);

// Compute norms
double norm2(const std::vector<double>& vec, size_t N);
double normInf(const std::vector<double>& vec, size_t N);

#endif // UTILS_HPP
