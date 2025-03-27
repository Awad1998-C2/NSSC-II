#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#include <mpi.h> // https://www-lb.open-mpi.org/doc/v4.1/

#include <chrono>
#include "utils.hpp"

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

int main(int argc, char** argv) {

  // first thing to do in an MPI program
  MPI_Init(&argc, &argv);

  // obtain own global rank
  int grk;
  MPI_Comm_rank(MPI_COMM_WORLD, &grk);

  // obtain number of global MPI processes
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  auto opts = parse(argc, argv);
  bool flag_one_process = (size == 1);

  // using a one-dimensional (1D) cartesian topology
    constexpr int n = 1; // one dimension

    // letting MPI find a proper select of the dimensions (obsolete for one-dimension)
    std::array<int, n> dims = {0};
    MPI_Dims_create(size, n, std::data(dims));

    // create a communicator for a 1D topology (with potential reordering of ranks)
    std::array<int, n> periodic = {false};
    int reorder = true; //??
    MPI_Comm comm;
    MPI_Cart_create(MPI_COMM_WORLD, n, std::data(dims), std::data(periodic), reorder, &comm);

    // obtain own rank in 1D communicator
    int rk;
    MPI_Comm_rank(comm, &rk);
    constexpr int displ = 1;
    std::array<int, n* 2> nb = {MPI_PROC_NULL, MPI_PROC_NULL}; 
    enum Direction : int { TOP = 0, BOTTOM = 1 };
    MPI_Cart_shift(comm, 0, displ, &nb[TOP], &nb[BOTTOM]);

    // constexpr int displ = 1;
    // std::array<int, n* 2> nb = {MPI_PROC_NULL, MPI_PROC_NULL};
    // enum Direction : int { LEFT = 0, RIGHT = 1 };
    // MPI_Cart_shift(comm, 0, displ, &nb[LEFT], &nb[RIGHT]);

    // obtain own coordinates
    std::array<int, n> coord = {-1};
    MPI_Cart_coords(comm, rk, n, std::data(coord));

    // setup own data domain: 3x3 row-major contiguous data:
    // rk , rk , rk
    // rk , rk , rk
    // rk , rk , rk
    // constexpr int N = 3;
    // std::array<double, N * N> data;
    // data.fill(rk);

    // calculating height data segment for given rank - 3 cases to consider 
    // int height = -1;
    size_t height = 0;
    if (rk == 0) height = (int)(opts.N/size) + (opts.N%size) + 1;
    else if (rk == size-1) height = (int)(opts.N/size) + 1;
    else height = (int)(opts.N/size) + 2;
    /* divide the segment evenly among all processors; spill over is assigned to rank 0; +2/+1 because of two/one ghost layer*/
    // special behaviour in case of one process
    if (flag_one_process) height = opts.N;

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

    // wait for all ranks to print initial state before communicating data
    MPI_Barrier(comm);

    // register MPI data types (here vectors) to send a row or a column
    MPI_Datatype row;
    // MPI_Type_vector(N, 1, 1, MPI_DOUBLE, &row);
    MPI_Type_vector(opts.N, 1, 1, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    // MPI_Datatype col;
    // MPI_Type_vector(opts.N, 1, N, MPI_DOUBLE, &col);
    // MPI_Type_commit(&col);

    //JACOBI update

    // solver update
    auto jacobi_iter = [N = opts.N, height, rk, size](const auto &xold, auto &xnew,
        bool residual = false) {
        auto h = 1.0 / (N - 1);
        auto h2 = h * h;
    if (rk == 0) {
        /* j=0: Dirichlet-Boundary; j=height-1: Ghost-Layer */
        // south boundary
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
        // all interior points
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
        /* j=0: Ghost-Layer; j=height-1: Dirichlet-Boundary */
        // North boundary
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
        /* j=0 and j=height-1: Ghost-Layer */
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



    // start clock
    const auto start = std::chrono::high_resolution_clock::now();

    // initialize x1 and x2 for Jacobi iteration
    // auto x1 = init();
    auto x1 = init(opts.N, opts.fix_west, opts.fix_east);
    auto x2 = x1;


    // send own middle column to both neighbors
    // recieve a column from left/right neighbor and store as right/left column

    // perform jacobi iterations
    // perform jacobi iterations
    enum Request : int { Bottom_SEND = 0, Top_SEND = 1, Bottom_RECV = 2, Top_RECV = 3 };
    enum Tag : int { Bottom_Msg = 0, Top_Msg = 1 };
    
    for (size_t iter = 0; iter <= opts.iters; ++iter) {
    
        if (size > 1) {  // Communication only needed if size > 1
            std::array<MPI_Request, 4> req;  // Declare per iteration
    
            // Corrected MPI Sends:
                MPI_Isend(&x1[opts.N], opts.N, MPI_DOUBLE, nb[TOP], Top_Msg, comm, &req[Top_SEND]); // send top row UP
                MPI_Isend(&x1[(height - 2)*opts.N], opts.N, MPI_DOUBLE, nb[BOTTOM], Bottom_Msg, comm, &req[Bottom_SEND]); // send bottom row DOWN

                // Corrected MPI Recvs:
                MPI_Irecv(&x1[0], opts.N, MPI_DOUBLE, nb[TOP], Bottom_Msg, comm, &req[Top_RECV]); // recv top ghost from above
                MPI_Irecv(&x1[(height - 1)*opts.N], opts.N, MPI_DOUBLE, nb[BOTTOM], Top_Msg, comm, &req[Bottom_RECV]); // recv bottom ghost from below
            // MPI_Sendrecv(&x1[opts.N], opts.N, MPI_DOUBLE, nb[TOP], Top_Msg,    // send to top
            //     &x1[0], opts.N, MPI_DOUBLE, nb[TOP], Bottom_Msg,      // recv from top into local boundary
            //     comm, MPI_STATUS_IGNORE);
   
            // // Communicate with BOTTOM neighbor
            // MPI_Sendrecv(&x1[(height - 2)*opts.N], opts.N, MPI_DOUBLE, nb[BOTTOM], Bottom_Msg,   // send to bottom
            //                 &x1[(height - 1)*opts.N], opts.N, MPI_DOUBLE, nb[BOTTOM], Top_Msg,      // recv from bottom into local boundary
            //                 comm, MPI_STATUS_IGNORE);
    
            // Wait for all communication to complete
            std::array<MPI_Status, 4> status;
            MPI_Waitall(4, req.data(), status.data());
        }
    
        // Perform computation only after the communication is completed
        if (size == 1) {  // Single process case
            jacobi_iter_1_process(x1, x2, opts.N, false); 
        } else {  // Multi-process case
            jacobi_iter(x1, x2, false);  // corrected function call
        }
    
        // Swap pointers for the next iteration
        std::swap(x1, x2);
    }
    
    // After completing iterations, compute residual
    if (size == 1) {
        jacobi_iter_1_process(x1, x2, opts.N, true);
    } else {
        jacobi_iter(x1, x2, true); // corrected function call
    }

    std::chrono::duration<double> dur = std::chrono::high_resolution_clock::now() - start;
    if (rk == 0) {
        std::cout << "Elapsed time: " << dur.count() << " s\n";
    }


    // NORMS start here 
// Gather residual into one global vector on rank 0
std::vector<int> recvcounts(size);
std::vector<int> displs(size);

// compute counts and displacements
for (int i = 0; i < size; ++i) {
    recvcounts[i] = (opts.N / size) * opts.N;
    if (static_cast<size_t>(i) < opts.N % size)
        recvcounts[i] += opts.N;  // distribute remainder
    displs[i] = (i > 0) ? (displs[i - 1] + recvcounts[i - 1]) : 0;
}

// local residual size
int local_size = (height - (rk == 0 || rk == size-1 ? 1 : 2)) * opts.N;
std::vector<double> global_residual;
if (rk == 0) global_residual.resize(opts.N * opts.N);

// Gather residual data (excluding ghost layers)
MPI_Gatherv(&x2[(rk == 0 ? 0 : opts.N)], local_size, MPI_DOUBLE,
            global_residual.data(), recvcounts.data(), displs.data(), MPI_DOUBLE,
            0, comm);

// Compute norms on rank 0
if (rk == 0) {
    double n2 = norm2(global_residual, opts.N);
    double nInf = normInf(global_residual, opts.N);

    std::cout << "norm2 = " << n2 << "\n";
    std::cout << "normInf = " << nInf << "\n";

    // 1. Write solution data using existing write function (to data/name.csv)
    write(global_residual, opts.N, opts.name); 

    // 2. Write benchmark data separately (for easy performance analysis)
    std::ofstream benchfile("data/" + opts.name + "_benchmark.csv", std::ios::app);

if (benchfile.tellp() == 0) {
    benchfile << "Resolution,Iterations,Processes,Runtime_sec,norm2,normInf\n";
}

benchfile << opts.N << ","
          << opts.iters << ","
          << size << ","
          << dur.count() << ","  // corrected line
          << n2 << ","
          << nInf << "\n";

benchfile.close();
}
// // Gather data and calculate norms on rank 0
// if (rk == 0) {
//     write(x1, opts.N, opts.name);
//     std::cout << "norm2 = ... (compute later)\n";
//     std::cout << "normInf = ... (compute later)\n";
// }

// Free resources
MPI_Type_free(&row);
MPI_Comm_free(&comm);

MPI_Barrier(MPI_COMM_WORLD);    
MPI_Finalize();

return 0;
}
