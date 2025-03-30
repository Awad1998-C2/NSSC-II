#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#include <chrono>
#include <cmath>
#include <cstddef>
#include <functional>
#include <stdexcept>
#include <string>
#include <tuple>

#include <mpi.h> // https://www-lb.open-mpi.org/doc/v4.1/

namespace program_options {

  struct Options {
    unsigned int mpi_mode;  
    std::string name;
    size_t N;
    size_t iters;
    double fix_west;
    double fix_east;  
    void print() const {
      std::printf("mpi_mode: %u\nD", mpi_mode);    
      std::printf("name: %s\n", name.c_str());
      std::printf("N: %zu\n", N);
      std::printf("iters: %zu\n", iters);
      std::printf("fix_west: %lf\n", fix_west);
      std::printf("fix_east: %lf\n", fix_east);    
    }
  };
  
  auto parse(int argc, char *argv[]) {
    if (argc != 7)
      throw std::runtime_error("unexpected number of arguments");
    Options opts;
    if (std::string(argv[1]) == std::string("1D"))
      opts.mpi_mode = 1;
    else if( std::string(argv[1]) == std::string("2D"))
      opts.mpi_mode = 2;
    else
     throw std::runtime_error("invalid parameter for mpi_mode (valid are '1D' and '2D')");
    opts.name = argv[2];
    if (std::sscanf(argv[3], "%zu", &opts.N) != 1 && opts.N >= 2)
      throw std::runtime_error("invalid parameter for N");
    if (std::sscanf(argv[4], "%zu", &opts.iters) != 1 && opts.iters != 0)
      throw std::runtime_error("invalid parameter for iters");
    if (std::sscanf(argv[5], "%lf", &opts.fix_west) != 1)
      throw std::runtime_error("invalid value for fix_west");
    if (std::sscanf(argv[6], "%lf", &opts.fix_east) != 1)
      throw std::runtime_error("invalid value for fix_east");  
    return opts;
  }
  
  } // namespace program_options
  

int main(int argc, char** argv) {

  // first thing to do in an MPI program
  MPI_Init(&argc, &argv);

  // obtain own global rank
  int grk;
  MPI_Comm_rank(MPI_COMM_WORLD, &grk);

  // obtain number of global MPI processes
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  // std::cout << "hello from global rank:" << grk << "/" << size << std::endl;
  // bool flag_one_process = (size == 1);
  


  { // using a one-dimensional (1D) cartesian topology

    constexpr int n = 1; // one dimension

    // letting MPI find a proper select of the dimensions (obsolete for one-dimension)
    std::array<int, n> dims = {0};
    MPI_Dims_create(size, n, std::data(dims));

    // create a communicator for a 1D topology (with potential reordering of ranks)
    std::array<int, n> periodic = {false};
    int reorder = true;
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD, n, std::data(dims), std::data(periodic), reorder, &comm);

    // obtain own rank in 1D communicator
    int rk;
    MPI_Comm_rank(comm, &rk);

    // obtain and store neighboring ranks in the 1D topology (left/right)
    constexpr int displ = 1;
    std::array<int, n* 2> nb = {MPI_PROC_NULL, MPI_PROC_NULL};
    // enum Direction : int { LEFT = 0, RIGHT = 1 };
    // MPI_Cart_shift(comm, 0, displ, &nb[LEFT], &nb[RIGHT]);
    enum Direction : int { Top = 0, Bottom = 1 };
    MPI_Cart_shift(comm, 0, displ, &nb[Bottom], &nb[Top]);

    // obtain own coordinates
    std::array<int, n> coord = {-1};
    MPI_Cart_coords(comm, rk, n, std::data(coord));

    auto opts = program_options::parse(argc, argv);

    // setup own data domain: 3x3 row-major contiguous data:
    // rk , rk , rk
    // rk , rk , rk
    // rk , rk , rk
    // constexpr int N = 3;
    // std::array<double, N * N> data;
    // data.fill(rk);


    // // wrapper for printing own data domain
    // auto print_own_domain = [&rk, &coord, &data](std::string filename) {
    //   std::ofstream file;
    //   file.open(filename);
    //   file << "coord=(" << coord[0] << ",) rank=" << rk << " :" << std::endl;
    //   for (const auto& col : {0, 1, 2}) {
    //     file << "[";
    //     for (const auto& row : {0, 1, 2})
    //       file << " " << data[row + N * col] << " ";
    //     file << "]" << std::endl;
    //   }
    // };

    // // print own data domain before commuincation
    // print_own_domain("domain_before_comm_rank_" + std::to_string(rk) + ".txt");

    // // wait for all ranks to print initial state before communicating data
    // MPI_Barrier(comm);

    // register MPI data types (here vectors) to send a row or a column
    MPI_Datatype row;
    // MPI_Type_vector(N, 1, 1, MPI_DOUBLE, &row);
    MPI_Type_vector(opts.N, 1, 1, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    // MPI_Datatype col;
    // MPI_Type_vector(N, 1, N, MPI_DOUBLE, &col);
    // MPI_Type_commit(&col);

    // how many rows each rank gets? assign remaining rows to rank 0, assign also ghost layers
    size_t height = 0;
    if (rk == 0) height = (int)(opts.N/size) + (opts.N%size) + 1;
    else if (rk == size-1) height = (int)(opts.N/size) + 1;
    else height = (int)(opts.N/size) + 2;


    // initial guess (0.0) with fixed values in west (-100) and east (100)
    auto init = [N = opts.N, W = opts.fix_west, E = opts.fix_east, height]() -> auto {
    // std::vector<double> res(N * N);
    std::vector<double> res(N * height);

    for (size_t j = 0; j < height; ++j)
      for (size_t i = 0; i < N; ++i) {
        res[i + j * N] = 0.0;
        if (i % N == 0)
          res[i + j * N] = W;
        if (i % N == N - 1)
          res[i + j * N] = E;
      }
    return res;
  };

  // solver update
  // auto jacobi_iter = [N = opts.N](const auto &xold, auto &xnew,
  //                                 bool residual = false) {
  auto jacobi_iter = [N = opts.N, height, rk, size](const auto &xold, auto &xnew,
                                  bool residual = false) {
    auto h = 1.0 / (N - 1);
    auto h2 = h * h;
    if (rk == 0) {
      size_t j = 0;
      for (size_t i = 1; i < N - 1; ++i) {
          auto w = xold[(i - 1) + (j)*N];
          auto e = xold[(i + 1) + (j)*N];
          auto n = xold[(i) + (j + 1) * N];
          auto s = n;
          auto c = xold[(i) + (j)*N];
          if (!residual)
          xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
          else
          xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4 * c);
      }
      for (size_t j = 1; j < (height-1); ++j) {
          for (size_t i = 1; i < N - 1; ++i) {
              auto w = xold[(i - 1) + (j)*N];
              auto e = xold[(i + 1) + (j)*N];
              auto n = xold[(i) + (j + 1) * N];
              auto s = xold[(i) + (j - 1) * N];
              auto c = xold[(i) + (j)*N];
              if (!residual)
              xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
              else
              xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
          }
      }
  }
  else if (rk == size-1) {
      size_t j = height - 1;
      for (size_t i = 1; i < N - 1; ++i) {
          auto w = xold[(i - 1) + (j)*N];
          auto e = xold[(i + 1) + (j)*N];
          auto s = xold[(i) + (j - 1) * N];
          auto n = s;
          auto c = xold[(i) + (j)*N];
          if (!residual)
          xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
          else
          xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4 * c);
      }
      for (size_t j = 1; j < height - 1; ++j) {
          for (size_t i = 1; i < N - 1; ++i) {
              auto w = xold[(i - 1) + (j)*N];
              auto e = xold[(i + 1) + (j)*N];
              auto n = xold[(i) + (j + 1) * N];
              auto s = xold[(i) + (j - 1) * N];
              auto c = xold[(i) + (j)*N];
              if (!residual)
              xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
              else
              xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
          }
          }
  }
  else {
      for (size_t j = 1; j < height - 1; ++j) {
          for (size_t i = 1; i < N - 1; ++i) {
              auto w = xold[(i - 1) + (j)*N];
              auto e = xold[(i + 1) + (j)*N];
              auto n = xold[(i) + (j + 1) * N];
              auto s = xold[(i) + (j - 1) * N];
              auto c = xold[(i) + (j)*N];
              if (!residual)
                  xnew[i + j * N] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
              else
                  xnew[i + j * N] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
          }
      }
  }
};

// write vector to csv
auto write = [N = opts.N, name = opts.name](const auto &x) -> auto {
  std::ofstream csv;
  csv.open(name + ".csv");
  for (size_t j = 0; j < N; ++j) {
    for (size_t i = 0; i < N - 1; ++i) {
      csv << x[i + j * N] << " ";
    }
    csv << x[(N - 1) + j * N];
    csv << "\n";
  }
  csv.close();
};

 // 2 norm
 auto norm2 = [N = opts.N](const auto &vec) -> auto {
  double sum = 0.0;
  for (size_t j = 0; j < N; ++j)
    for (size_t i = 1; i < (N - 1); ++i)
      sum += vec[i + j * N] * vec[i + j * N];

  return std::sqrt(sum);
};

// Inf norm
auto normInf = [N = opts.N](const auto &vec) -> auto {
  double max = 0.0;
  for (size_t j = 0; j < N; ++j)
    for (size_t i = 1; i < (N - 1); ++i)
      max = std::fabs(vec[i + j * N]) > max ? std::fabs(vec[i + j * N]) : max;
  return max;
};


    // start clock
    const auto start = std::chrono::high_resolution_clock::now();

    // initialize x1 and x2
    auto x1 = init();
    auto x2 = x1;



    // send own middle column to both neighbors
    // recieve a column from left/right neighbor and store as right/left column
    // enum Request : int { LEFT_SEND = 0, RIGHT_SEND = 1, LEFT_RECV = 2, RIGHT_RECV = 3 };
    // enum Tag : int { LEFT_Msg = 0, RIGHT_Msg = 1 };

    enum Request : int { BOTTOM_SEND = 0, TOP_SEND = 1, BOTTOM_RECV = 2, TOP_RECV = 3 };
    enum Tag : int { BOTTOM_Msg = 0, TOP_Msg = 1 };
    for (size_t iter = 0; iter <= opts.iters; ++iter) {
      jacobi_iter(x1, x2);
      std::swap(x1, x2);


    std::array<MPI_Request, n * 2 * 2> req;
    // MPI_Isend(&data[1], 1, col, nb[LEFT], LEFT_Msg, comm, &req[LEFT_SEND]);
    // MPI_Isend(&data[1], 1, col, nb[RIGHT], RIGHT_Msg, comm, &req[RIGHT_SEND]);
    // MPI_Irecv(&data[0], 1, col, nb[LEFT], RIGHT_Msg, comm, &req[LEFT_RECV]);
    // MPI_Irecv(&data[2], 1, col, nb[RIGHT], LEFT_Msg, comm, &req[RIGHT_RECV]);

      MPI_Isend(&x1[(1)*opts.N], 1, row, nb[Bottom], BOTTOM_Msg, comm, &req[BOTTOM_SEND]);
      MPI_Isend(&x1[(height-2)*opts.N], 1, row, nb[Top], TOP_Msg, comm, &req[TOP_SEND]);
      MPI_Irecv(&x1[0], 1, row, nb[Bottom], TOP_Msg, comm, &req[BOTTOM_RECV]);
      MPI_Irecv(&x1[(height-1)*opts.N], 1, row, nb[Top], BOTTOM_Msg, comm, &req[TOP_RECV]);
      
    // wait/block for/until all four communication calls to finish
    std::array<MPI_Status, n * 2 * 2> status;
    MPI_Waitall(n * 2 * 2, std::data(req), std::data(status));

    // // residual calculation
    // jacobi_iter(x1, x2, true);

    // //gather
    // //use MPI_Gather to collect all partial results
    // auto gather_all_parts = [N = opts.N, rk, size, height, comm, row](const auto& vec) -> auto {
    //   // initialising receive buffer
    //   std::vector<double> recvbuf;
    //   if (rk == 0) {
    //     recvbuf = vec;
    //     recvbuf.resize(N*N, -1);
    //   }
    //   // gather different parts of different processes into process rk=0
    //   if (rk == 0)
    //     MPI_Gather(&vec[(N%size)*N], height-1-(N%size), row, std::data(recvbuf)+((N%size)*N), height-1-(N%size), row, 0, comm);
    //   else if (rk == size-1)
    //     MPI_Gather(&vec[(1)*N], height-1, row, std::data(recvbuf)+((N%size)*N), height-1, row, 0, comm);
    //   else
    //     MPI_Gather(&vec[(1)*N], height-2, row, std::data(recvbuf)+((N%size)*N), height-2, row, 0, comm);

    //   return recvbuf;
    // };

    // std::vector<double> solution;
    // std::vector<double> residual;

    // // perform gather for solution and residual
    // solution = gather_all_parts(x1);
    // residual = gather_all_parts(x2);

    // residual calculation
jacobi_iter(x1, x2, true);

// Helper lambda for Gatherv
auto gather_all_parts_v = [N = opts.N, rk, size, height, comm](const auto& vec) {
  std::vector<int> recvcounts(size), displs(size);
  int local_rows = height - ((rk == 0 || rk == size - 1) ? 1 : 2);

  // All processes send their number of rows to root
  MPI_Gather(&local_rows, 1, MPI_INT, recvcounts.data(), 1, MPI_INT, 0, comm);

  if (rk == 0) {
    for (int i = 0, sum = 0; i < size; ++i) {
      recvcounts[i] *= N;       // Number of elements per process
      displs[i] = sum;          // Offset in global array
      sum += recvcounts[i];
    }
  }

  // Prepare send buffer (excluding ghost layers)
  int send_offset = (rk == 0) ? 0 : N;
  const double* sendbuf = vec.data() + send_offset;

  // Allocate receive buffer at root
  std::vector<double> recvbuf;
  if (rk == 0)
    recvbuf.resize(N * N, -1);

  // Perform MPI_Gatherv
  MPI_Gatherv(sendbuf, local_rows * N, MPI_DOUBLE,
              recvbuf.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
              0, comm);

  return recvbuf;
};

// perform gather for solution and residual
auto solution = gather_all_parts_v(x1);
auto residual = gather_all_parts_v(x2);


    const auto end = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);

    if (rk == 0) {
      write(solution);
      std::cout << "  norm2 = " << norm2(residual) << std::endl;
      std::cout << "normInf = " << normInf(residual) << std::endl;
      std::cout << "Runtime: " << duration.count() << " microseconds" << std::endl;
    }

    // print own domain data again, now this should look likes this:
    // rk-1 , rk , rk+1
    // rk-1 , rk , rk+1
    // rk-1 , rk , rk+1
    // print_own_domain("domain_after_comm_rank_" + std::to_string(rk) + ".txt");

    // free 1d communicator
    // MPI_Comm_free(&comm);
  }

    MPI_Comm_free(&comm);


  // MPI_Barrier(MPI_COMM_WORLD);    
  // call the MPI final cleanup routine
  MPI_Finalize();

  return 0;
}
}
