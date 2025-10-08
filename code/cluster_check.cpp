#include <mpi.h>

#include <iostream>

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int nprocs;
  int myrank;
  MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  char processor_name[MPI_MAX_PROCESSOR_NAME];
  int name_len;
  MPI_Get_processor_name(processor_name, &name_len);

  for (int i = 0; i < nprocs; ++i) {
    if (i == myrank)
      printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, myrank, nprocs);
    MPI_Barrier(MPI_COMM_WORLD);
  }

  MPI_Finalize();
  return 0;
}