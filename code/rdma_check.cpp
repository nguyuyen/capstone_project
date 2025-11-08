#include <mpi.h>

#include <iostream>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int nprocs;
  int myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);

  int n = myrank;
  MPI_Win w;
  MPI_Win_create(&n, sizeof(int), sizeof(int), MPI_INFO_NULL, MPI_COMM_WORLD, &w);

  MPI_Win_fence(0, w);
  MPI_Win_lock_all(0, w);

  MPI_Barrier(MPI_COMM_WORLD);

  int temp = myrank;

  int result = 0;

  for (int i = 0; i < 100; ++i) {
    MPI_Get_accumulate(NULL, 0, MPI_INT, &result, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_NO_OP, w);
    MPI_Win_flush(0, w);
    MPI_Accumulate(&temp, 1, MPI_INT, 0, 0, 1, MPI_INT, MPI_REPLACE, w);
    MPI_Win_flush(0, w);
  }

  MPI_Win_unlock_all(w);
  MPI_Win_free(&w);

  MPI_Finalize();
  return 0;
}