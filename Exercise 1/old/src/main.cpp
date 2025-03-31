#include "utils.hpp"
#include <iostream>
#include <stdexcept>
#include <string>

struct Options {
  unsigned int mpi_mode;  
  std::string name;
  size_t N;
  size_t iters;
  double fix_west;
  double fix_east;  
  void print() const {
    std::printf("mpi_mode: %u\n", mpi_mode);    
    std::printf("name: %s\n", name.c_str());
    std::printf("N: %zu\n", N);
    std::printf("iters: %zu\n", iters);
    std::printf("fix_west: %lf\n", fix_west);
    std::printf("fix_east: %lf\n", fix_east);    
  }
};

Options parse(int argc, char *argv[]) {
  if (argc != 7)
    throw std::runtime_error("unexpected number of arguments");
  
  Options opts;
  if (std::string(argv[1]) == "1D")
    opts.mpi_mode = 1;
  else if (std::string(argv[1]) == "2D")
    opts.mpi_mode = 2;
  else
    throw std::runtime_error("invalid parameter for mpi_mode (valid are '1D' and '2D')");
  
  opts.name = argv[2];

  if (std::sscanf(argv[3], "%zu", &opts.N) != 1 || opts.N < 2)
    throw std::runtime_error("invalid parameter for N");
  if (std::sscanf(argv[4], "%zu", &opts.iters) != 1 || opts.iters == 0)
    throw std::runtime_error("invalid parameter for iters");
  if (std::sscanf(argv[5], "%lf", &opts.fix_west) != 1)
    throw std::runtime_error("invalid value for fix_west");
  if (std::sscanf(argv[6], "%lf", &opts.fix_east) != 1)
    throw std::runtime_error("invalid value for fix_east");  

  return opts;
}

int main(int argc, char *argv[]) try {
    auto opts = parse(argc, argv);
    opts.print();

    auto x1 = init(opts.N, opts.fix_west, opts.fix_east);
    auto x2 = x1;

    for (size_t iter = 0; iter <= opts.iters; ++iter) {
        jacobi_iter_1_process(x1, x2, opts.N);
        std::swap(x1, x2);
    }

    write(x2, opts.N, opts.name);
    jacobi_iter_1_process(x1, x2, opts.N, true);

    std::cout << "  norm2 = " << norm2(x2, opts.N) << '\n';
    std::cout << "normInf = " << normInf(x2, opts.N) << '\n';

    return EXIT_SUCCESS;
} catch (const std::exception &e) {
    std::cerr << e.what() << '\n';
    return EXIT_FAILURE;
}

