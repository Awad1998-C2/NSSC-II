#include "utils.hpp"
#include <vector>
#include <cmath>
#include <fstream>

std::vector<double> init(size_t N, double fix_west, double fix_east) {
    std::vector<double> data(N * N, 0.0);

    // Set west (left) boundary
    for (size_t j = 0; j < N; ++j)
        data[j * N + 0] = fix_west;

    // Set east (right) boundary
    for (size_t j = 0; j < N; ++j)
        data[j * N + (N - 1)] = fix_east;

    return data;
}


void jacobi_iter_1_process(const std::vector<double>& xold, std::vector<double>& xnew, size_t N, bool residual) {
    auto h = 1.0 / (N - 1);
    auto h2 = h * h;

    for (size_t j = 1; j < N - 1; ++j) {
        for (size_t i = 1; i < N - 1; ++i) {
            auto w = xold[(i - 1) + j*N];
            auto e = xold[(i + 1) + j*N];
            auto n = xold[i + (j + 1)*N];
            auto s = xold[i + (j - 1)*N];
            auto c = xold[i + j*N];
            if (!residual)
                xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
            else
                xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
        }
    }

    // South boundary
    size_t j = 0;
    for (size_t i = 1; i < N - 1; ++i) {
        auto w = xold[(i - 1) + j*N];
        auto e = xold[(i + 1) + j*N];
        auto n = xold[i + (j + 1)*N];
        auto s = n;
        auto c = xold[i + j*N];
        if (!residual)
            xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
        else
            xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4 * c);
    }

    // North boundary
    j = N - 1;
    for (size_t i = 1; i < N - 1; ++i) {
        auto w = xold[(i - 1) + j*N];
        auto e = xold[(i + 1) + j*N];
        auto s = xold[i + (j - 1)*N];
        auto n = s;
        auto c = xold[i + j*N];
        if (!residual)
            xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
        else
            xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4 * c);
    }
}

void write(const std::vector<double>& x, size_t N, const std::string& name) {
    std::ofstream csv("data/" + name + ".csv");  // <-- correct path
    for (size_t j = 0; j < N; ++j) {
        for (size_t i = 0; i < N - 1; ++i)
            csv << x[i + j * N] << " ";
        csv << x[(N - 1) + j * N] << '\n';
    }
}

double norm2(const std::vector<double>& vec, size_t N) {
    double sum = 0.0;
    for (size_t j = 0; j < N; ++j)
        for (size_t i = 1; i < N - 1; ++i)
            sum += vec[i + j * N] * vec[i + j * N];
    return std::sqrt(sum);
}

double normInf(const std::vector<double>& vec, size_t N) {
    double max = 0.0;
    for (size_t j = 0; j < N; ++j)
        for (size_t i = 1; i < N - 1; ++i)
            max = std::fabs(vec[i + j * N]) > max ? std::fabs(vec[i + j * N]) : max;
    return max;
}
