#include <algorithm>
#include <array>
#include <fstream>
#include <iostream>
#include <iterator>
#include <sstream>
#include <vector>

#include <mpi.h> // https://www-lb.open-mpi.org/doc/v4.1/

int main(int argc, char** argv) {

  // first thing to do in an MPI program
  MPI_Init(&argc, &argv);

  // obtain own global rank
  int grk;
  MPI_Comm_rank(MPI_COMM_WORLD, &grk);

  // obtain number of global MPI processes
  int size;
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  std::cout << "hello from global rank:" << grk << "/" << size << std::endl;


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
    enum Direction : int { LEFT = 0, RIGHT = 1 };
    MPI_Cart_shift(comm, 0, displ, &nb[LEFT], &nb[RIGHT]);

    // obtain own coordinates
    std::array<int, n> coord = {-1};
    MPI_Cart_coords(comm, rk, n, std::data(coord));

    // setup own data domain: 3x3 row-major contiguous data:
    // rk , rk , rk
    // rk , rk , rk
    // rk , rk , rk
    constexpr int N = 3;
    std::array<double, N * N> data;
    data.fill(rk);

    // wrapper for printing own data domain
    auto print_own_domain = [&rk, &coord, &data](std::string filename) {
      std::ofstream file;
      file.open(filename);
      file << "coord=(" << coord[0] << ",) rank=" << rk << " :" << std::endl;
      for (const auto& col : {0, 1, 2}) {
        file << "[";
        for (const auto& row : {0, 1, 2})
          file << " " << data[row + N * col] << " ";
        file << "]" << std::endl;
      }
    };

    // print own data domain before commuincation
    print_own_domain("domain_before_comm_rank_" + std::to_string(rk) + ".txt");

    // wait for all ranks to print initial state before communicating data
    MPI_Barrier(comm);

    // register MPI data types (here vectors) to send a row or a column
    MPI_Datatype row;
    MPI_Type_vector(N, 1, 1, MPI_DOUBLE, &row);
    MPI_Type_commit(&row);
    MPI_Datatype col;
    MPI_Type_vector(N, 1, N, MPI_DOUBLE, &col);
    MPI_Type_commit(&col);

    // send own middle column to both neighbors
    // recieve a column from left/right neighbor and store as right/left column
    enum Request : int { LEFT_SEND = 0, RIGHT_SEND = 1, LEFT_RECV = 2, RIGHT_RECV = 3 };
    enum Tag : int { LEFT_Msg = 0, RIGHT_Msg = 1 };
    std::array<MPI_Request, n * 2 * 2> req;
    MPI_Isend(&data[1], 1, col, nb[LEFT], LEFT_Msg, comm, &req[LEFT_SEND]);
    MPI_Isend(&data[1], 1, col, nb[RIGHT], RIGHT_Msg, comm, &req[RIGHT_SEND]);
    MPI_Irecv(&data[0], 1, col, nb[LEFT], RIGHT_Msg, comm, &req[LEFT_RECV]);
    MPI_Irecv(&data[2], 1, col, nb[RIGHT], LEFT_Msg, comm, &req[RIGHT_RECV]);

    // wait/block for/until all four communication calls to finish
    std::array<MPI_Status, n * 2 * 2> status;
    MPI_Waitall(n * 2 * 2, std::data(req), std::data(status));

    // print own domain data again, now this should look likes this:
    // rk-1 , rk , rk+1
    // rk-1 , rk , rk+1
    // rk-1 , rk , rk+1
    print_own_domain("domain_after_comm_rank_" + std::to_string(rk) + ".txt");

    // free 1d communicator
    MPI_Comm_free(&comm);
  }

  MPI_Barrier(MPI_COMM_WORLD);    
  // call the MPI final cleanup routine
  MPI_Finalize();

  return 0;
}
