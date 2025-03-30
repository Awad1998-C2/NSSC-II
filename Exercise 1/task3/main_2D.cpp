#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>
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
  
// between 1-40 processes; fallback to 1D
bool check_if_prime(int n) {
    for (int i=2; i<n; i++) {
        if (n%i==0) {
            return false;
        }
    }
    return true;
}

// bool check_if_prime(int n) {
//   if (n <= 1) return false;         // 0 and 1 are not prime
//   if (n == 2) return true;          // 2 is prime
//   if (n % 2 == 0) return false;     // eliminate even numbers

//   for (int i = 3; i * i <= n; i += 2) {
//       if (n % i == 0) return false;
//   }
//   return true;
// }

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

  { // using a two-dimensional (2D) cartesian topology

    // prime fallback
    // bool fallback_1d = check_if_prime(size);

    constexpr int n = 2; // one dimension

    // letting MPI find a proper select of the dimensions (obsolete for one-dimension)
    std::array<int, n> dims = {0, 0};
    MPI_Dims_create(size, n, std::data(dims));

    // number of rows and column may not be eq eg 2x3
    int num1 = dims[0];
    int num2 = dims[1];

    // create a communicator for a 2D topology (with potential reordering of ranks)
    std::array<int, n> periodic = {false, false};
    int reorder = true;
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD, n, std::data(dims), std::data(periodic), reorder, &comm);

    // obtain own rank in 2D communicator
    int rk;
    MPI_Comm_rank(comm, &rk);

    // obtain and store neighboring ranks in the 2D topology (left/right)
    constexpr int displ = 1;
    std::array<int, n * 2> nb = {MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL, MPI_PROC_NULL};
    enum Direction : int { LEFT = 0, RIGHT = 1, TOP = 2, BOTTOM = 3 };
    MPI_Cart_shift(comm, 0, displ, &nb[BOTTOM], &nb[TOP]);
    MPI_Cart_shift(comm, 1, displ, &nb[LEFT], &nb[RIGHT]);

    // obtain own coordinates
    std::array<int, n> coord = {-1, -1};
    MPI_Cart_coords(comm, rk, n, std::data(coord));

    // // setup own data domain: 4x4 row-major contiguous data:
    // // rk , rk , rk, rk
    // // rk , rk , rk, rk
    // // rk , rk , rk, rk
    // // rk , rk , rk, rk
    // constexpr int N = 4;
    // std::array<double, N * N> data;
    // data.fill(rk);

    // instead of filling the data with rk, we fill it with the coordinates
    auto opts = program_options::parse(argc, argv);

    // // wrapper for printing own data domain
    // auto print_own_domain = [&rk, &coord, &data](std::string filename) {
    //   std::ofstream file;
    //   file.open(filename);
    //   file << "coord=(" << coord[0] << "," << coord[1] << ") rank=" << rk << " :" << std::endl;
    //   for (const auto& col : {0, 1, 2, 3}) {
    //     file << "[";
    //     for (const auto& row : {0, 1, 2, 3})
    //       file << " " << data[row + N * col] << " ";
    //     file << "]" << std::endl;
    //   }
    // };

    // print own data domain before commuincation
    // print_own_domain("domain_before_comm_rank_" + std::to_string(rk) + ".txt");

        // calculate hight and width of segment
        int height = -1;
        int width = -1;
        if (coord[0] == 0 && coord[1] == 0) {
            height = ( (int)(opts.N/num1) ) + (opts.N%num1);
            width = ( (int)(opts.N/num2) ) + (opts.N%num2);
        }
        else if (coord[0] == 0) {
            height = ( (int)(opts.N/num1) ) + (opts.N%num1);
            width = ( (int)(opts.N/num2) );
        }
        else if (coord[1] == 0) {
            height = ( (int)(opts.N/num1) );
            width = ( (int)(opts.N/num2) ) + (opts.N%num2);       
        }
        else {
            height = ( (int)(opts.N/num1) );
            width = ( (int)(opts.N/num2) );
        }
        // ghost layer
        if ( (coord[0]==0 && coord[1]==0) || (coord[0]==0 && coord[1]==num2-1) || (coord[0]==num1-1 && coord[1]==0) || (coord[0]==num1-1 && coord[1]==num2-1) ) {
            // corner pieces
            height += 1;
            width += 1;
        }
        else if ( (coord[0]==0) || (coord[0]==num1-1) ) {
            // horizontal edges
            width += 2;
            height += 1;
        }
        else if ( (coord[1]==0) || (coord[1]==num2-1) ) {
            // vertical edges
            width += 1;
            height += 2;
        }
        else {
            // inside
            width += 2;   
            height += 2;
        }
        if (flag_1d) {
            // this is equivalent to width--; this removes the ghost layer in horizontal direction
            width = opts.N;
        }

    // wait for all ranks to print initial state before communicating data
    MPI_Barrier(comm);

    // register MPI data types (here vectors) to send a row or a column
    // MPI_Datatype row;
    // MPI_Type_vector(N - 2, 1, 1, MPI_DOUBLE, &row);
    // MPI_Type_commit(&row);
    // MPI_Datatype col;
    // MPI_Type_vector(N - 2, 1, N, MPI_DOUBLE, &col);
    // MPI_Type_commit(&col);
    MPI_Datatype row;
    MPI_Type_vector(width, 1, 1, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    MPI_Datatype col;
    MPI_Type_vector(height, 1, width, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);

    //for a global row
    MPI_Datatype long_row;
    MPI_Type_vector(opts.N, 1, 1, MPI_DOUBLE, &long_row);
    MPI_Type_commit(&long_row);

       // initial guess (0.0) with fixed values in west and east
       auto init = [num1, num2, coord, width, height, W = opts.fix_west, E = opts.fix_east]() -> auto {
        std::vector<double> res(width * height);
        for (size_t j = 0; j < height; ++j)
            for (size_t i = 0; i < width; ++i) {
                res[i + j * width] = 0.0;
                if (i % width == 0)
                res[i + j * width] = (coord[1] == 0) ? W : 0;
                if (i % width == width - 1)
                res[i + j * width] = (coord[1] == num2-1) ? E : 0;
            }
        return res;
    };

    // solver update for only one process
    auto jacobi_iter_one_process = [N = opts.N](const auto &xold, auto &xnew,
                                    bool residual = false) {
        auto h = 1.0 / (N - 1);
        auto h2 = h * h;
        // all interior points
        for (size_t j = 1; j < N - 1; ++j) {
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
        // isolating south boundary
        {
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
        }
        // isolating north boundary
        {
        size_t j = N - 1;
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
        }
    };

    // solver update for more than one process
    auto jacobi_iter = [width, height, coord, num1, num2, N=opts.N](const auto &xold, auto &xnew,
                                    bool residual = false) {
        auto h = 1.0 / (N - 1);
        auto h2 = h * h;
        if (coord[0]==0) {
            /* j=0: Neumann-Boundary; j=height-1: Ghost-Layer */
            // south boundary
            size_t j = 0;
            for (size_t i = 1; i < width - 1; ++i) {
                auto w = xold[(i - 1) + (j)*width];
                auto e = xold[(i + 1) + (j)*width];
                auto n = xold[(i) + (j + 1) * width];
                auto s = n;
                auto c = xold[(i) + (j)*width];
                if (!residual)
                xnew[i + j * width] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
                else
                xnew[i + j * width] = (-1.0 / h2) * (w + e + n + s - 4 * c);
            }
            // all interior points
            for (size_t j = 1; j < (height-1); ++j) {
                for (size_t i = 1; i < width - 1; ++i) {
                    auto w = xold[(i - 1) + (j)*width];
                    auto e = xold[(i + 1) + (j)*width];
                    auto n = xold[(i) + (j + 1) * width];
                    auto s = xold[(i) + (j - 1) * width];
                    auto c = xold[(i) + (j)*width];
                    if (!residual)
                    xnew[i + j * width] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
                    else
                    xnew[i + j * width] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
                }
            }
        }
        else if (coord[0] == num1-1) {
            /* j=0: Ghost-Layer; j=height-1: Dirichlet-Boundary */
            // North boundary
            size_t j = height - 1;
            for (size_t i = 1; i < width - 1; ++i) {
                auto w = xold[(i - 1) + (j)*width];
                auto e = xold[(i + 1) + (j)*width];
                auto s = xold[(i) + (j - 1) * width];
                auto n = s;
                auto c = xold[(i) + (j)*width];
                if (!residual)
                xnew[i + j * width] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
                else
                xnew[i + j * width] = (-1.0 / h2) * (w + e + n + s - 4 * c);
            }
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < width - 1; ++i) {
                    auto w = xold[(i - 1) + (j)*width];
                    auto e = xold[(i + 1) + (j)*width];
                    auto n = xold[(i) + (j + 1) * width];
                    auto s = xold[(i) + (j - 1) * width];
                    auto c = xold[(i) + (j)*width];
                    if (!residual)
                    xnew[i + j * width] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
                    else
                    xnew[i + j * width] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
                }
                }
        }
        else {
            /* j=0 and j=height-1: Ghost-Layer */
            for (size_t j = 1; j < height - 1; ++j) {
                for (size_t i = 1; i < width - 1; ++i) {
                    auto w = xold[(i - 1) + (j)*width];
                    auto e = xold[(i + 1) + (j)*width];
                    auto n = xold[(i) + (j + 1) * width];
                    auto s = xold[(i) + (j - 1) * width];
                    auto c = xold[(i) + (j)*width];
                    if (!residual)
                        xnew[i + j * width] = (- (-1.0 / h2) * (w + e + n + s)) * h2 / 4.0;
                    else
                        xnew[i + j * width] = (-1.0 / h2) * (w + e + n + s - 4.0 * c);
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

    // print segment of a specific rank
    auto print_vec = [rk](auto vec, int height, int width, int rank=0) {
        if (rk == rank) {
            for (int j=height-1; j>=0; j--) {
                for (int i=0; i<width-1; i++) {
                    std::cout << vec[i + width*j] << "\t";
                }
                std::cout << vec[width-1 + width*j] << std::endl;
            }
        }
    };
    
// start clock
    const auto start = std::chrono::high_resolution_clock::now();

    // initialize x1 and x2
    auto x1 = init();
    auto x2 = x1;


    for (size_t iter = 0; iter <= opts.iters; ++iter) {
      jacobi_iter(x1, x2);
      std::swap(x1, x2);
    // send/recv to all four neighbors
    std::array<MPI_Request, n * 2 * 2> req;

    // // send own middle column to left; recv the middle column from right
    // MPI_Isend(&data[N + 1], 1, col, nb[LEFT], LEFT, comm, &req[0]);
    // MPI_Irecv(&data[N + N - 1], 1, col, nb[RIGHT], LEFT, comm, &req[1]);

    // // send own middle column to right; recv the middle column from left
    // MPI_Isend(&data[N + 1], 1, col, nb[RIGHT], RIGHT, comm, &req[2]);
    // MPI_Irecv(&data[N], 1, col, nb[LEFT], RIGHT, comm, &req[3]);

    // // send own middle row to bottom; recv the middle row from top
    // MPI_Isend(&data[N + 1], 1, row, nb[BOTTOM], BOTTOM, comm, &req[4]);
    // MPI_Irecv(&data[N * N - N + 1], 1, row, nb[TOP], BOTTOM, comm, &req[5]);

    // // send own middle row to top; recv the middle row from bottom
    // MPI_Isend(&data[N + 1], 1, row, nb[TOP], TOP, comm, &req[6]);
    // MPI_Irecv(&data[+1], 1, row, nb[BOTTOM], TOP, comm, &req[7]);

    // send to the left
    MPI_Isend(&x1[1], 1, col, nb[LEFT], LEFT, comm, &req[0]);
    MPI_Irecv(&x1[width-1], 1, col, nb[RIGHT], LEFT, comm, &req[1]);

    // send to the right
    MPI_Isend(&x1[width-2], 1, col, nb[RIGHT], RIGHT, comm, &req[2]);
    MPI_Irecv(&x1[0], 1, col, nb[LEFT], RIGHT, comm, &req[3]);

    // send to the bottom
    MPI_Isend(&x1[(1)*width], 1, row, nb[BOTTOM], BOTTOM, comm, &req[4]);
    MPI_Irecv(&x1[(height-1)*width], 1, row, nb[TOP], BOTTOM, comm, &req[5]);

    // send to the top
    MPI_Isend(&x1[(height-2)*width], 1, row, nb[TOP], TOP, comm, &req[6]);
    MPI_Irecv(&x1[0], 1, row, nb[BOTTOM], TOP, comm, &req[7]);


    // wait/block for/until all four communication calls to finish
    std::array<MPI_Status, n * 2 * 2> status;
    MPI_Waitall(n * 2 * 2, std::data(req), std::data(status));
    }
    // print own domain data again, now this should look likes this:
    // rk   , TOP    , TOP    , rk
    // LEFT , rk     , rk     , RIGHT
    // LEFT , rk     , rk     , RIGHT
    // rk   , BOTTOM , BOTTOM , rk
    // print_own_domain("domain_after_comm_rank_" + std::to_string(rk) + ".txt");

    // wrait for all processes in the 2D communicator to reach here
    // MPI_Barrier(comm);

    // close and print the shared cout to the console
    // MPI_File_close(&shared_cout);

    jacobi_iter(x1, x2, true);
    // transpose
    auto transpose = [](const auto& vec, int height, int width)->auto {
      std::vector<double> transposed_vec(vec.size());
      for (int j=0; j<height; j++) {
          for (int i=0; i<width; i++) {
              transposed_vec[j + i*height] = vec[i + j*width];
          }
      }
      return transposed_vec;
  };

  // create communicators for row and column
  MPI_Comm row_comm;
  MPI_Comm_split(comm, coord[0], coord[1], &row_comm);
  MPI_Comm col_comm;
  MPI_Comm_split(comm, coord[1], coord[0], &col_comm);

  // use MPI_Gather to perform vertical gather
  auto gather_all_parts_vertical = [N = opts.N, coord, height, width, col_comm, row, num1](const auto& vec) -> auto {
    // initialising receive buffer
    std::vector<double> recvbuf;
    if (coord[0] == 0) {
      recvbuf = vec;
      recvbuf.resize(N*width, -1);
    }
    // gather different parts of different processes into processes with column-rank 0
    if (coord[0]==0)
      // processes at the bottom
      MPI_Gather(&vec[(N%num1)*width], height-1-(N%num1), row, std::data(recvbuf)+((N%num1)*width), height-1-(N%num1), row, 0, col_comm);
    else if (coord[0]==num1-1)
      // processes at the top
      MPI_Gather(&vec[(1)*width], height-1, row, std::data(recvbuf)+((N%num1)*width), height-1, row, 0, col_comm);
    else
      MPI_Gather(&vec[(1)*width], height-2, row, std::data(recvbuf)+((N%num1)*width), height-2, row, 0, col_comm);

    return recvbuf;
  };

  // use MPI_Gather to perform horizontal gather
  auto gather_all_parts_horizontal = [N = opts.N, coord, width, row_comm, long_row, num2](const auto& vec) -> auto {
      // initialising receive buffer
      std::vector<double> recvbuf;
      if (coord[0]==0 && coord[1]==0) {
          recvbuf = vec;
          recvbuf.resize(N*N, -1);
      }
      // gather different parts of different processes into process rk 0
      if (coord[1]==0)
          // process with rk 0
          MPI_Gather(&vec[(N%num2)*N], width-1-(N%num2), long_row, std::data(recvbuf)+((N%num2)*N), width-1-(N%num2), long_row, 0, row_comm);
      else if (coord[1]==num2-1)
          // process most right at the bottom
          MPI_Gather(&vec[(1)*N], width-1, long_row, std::data(recvbuf)+((N%num2)*N), width-1, long_row, 0, row_comm);
      else
          MPI_Gather(&vec[(1)*N], width-2, long_row, std::data(recvbuf)+((N%num2)*N), width-2, long_row, 0, row_comm);

      return recvbuf;
  };

  // use MPI_Gather to collect all partial results in case of prime number --> 1d
  auto gather_all_parts_1d = [N = opts.N, rk, size, height, comm, row](const auto& vec) -> auto {
    // initialising receive buffer
    std::vector<double> recvbuf;
    if (rk == 0) {
      recvbuf = vec;
      recvbuf.resize(N*N, -1);
    }
    // gather different parts of different processes into process rk=0
    if (rk == 0)
      MPI_Gather(&vec[(N%size)*N], height-1-(N%size), row, std::data(recvbuf)+((N%size)*N), height-1-(N%size), row, 0, comm);
    else if (rk == size-1)
      MPI_Gather(&vec[(1)*N], height-1, row, std::data(recvbuf)+((N%size)*N), height-1, row, 0, comm);
    else
      MPI_Gather(&vec[(1)*N], height-2, row, std::data(recvbuf)+((N%size)*N), height-2, row, 0, comm);

    return recvbuf;
  };

  std::vector<double> solution;
  std::vector<double> residual;
  if (flag_1d) {
      // perform gather for solution and residual
      solution = gather_all_parts_1d(x1);
      residual = gather_all_parts_1d(x2);
  }
  else if(flag_one_process) {
      // no action necessary
  }
  else {
      // perform vertical gather
      solution = gather_all_parts_vertical(x1);
      if (coord[0]==0) {
          solution = gather_all_parts_horizontal(transpose(solution, opts.N, width));
      }
      // perform horizontal gather
      residual = gather_all_parts_vertical(x2);
      if (coord[0]==0) {
          residual = gather_all_parts_horizontal(transpose(residual, opts.N, width));
      }
      // perform final transposition
      if (rk==0) {
          solution = transpose(solution, opts.N, opts.N);
          residual = transpose(residual, opts.N, opts.N);
      }
  }

  // stop clock and calculate duration
  const auto end = std::chrono::high_resolution_clock::now();
  auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end - start);
  
  // write solution to file and print norms to console
  if (rk == 0) {
    write(solution);
    std::cout << "  norm2 = " << norm2(residual) << std::endl;
    std::cout << "normInf = " << normInf(residual) << std::endl;
    std::cout << "Runtime: " << duration.count() << " microseconds" << std::endl;
  }


    // free 2D communicator
    // MPI_Barrier(MPI_COMM_WORLD);    
    // call the MPI final cleanup routine
    MPI_Comm_free(&col_comm);
    MPI_Comm_free(&row_comm);
    MPI_Comm_free(&comm);
  }


  MPI_Finalize();

  return 0;
}
