#define MEM_CHECK

#include <mpi.h>
#include <unistd.h>

#include <fstream>
#include <iostream>
#include <random>

#include "hash_map.h"
#include "test_bench.h"

// #define AFRAID

int main(int argc, char** argv) {
  int total_op = atoi(argv[1]);
  int insert_weight = atoi(argv[2]);
  int get_weight = atoi(argv[3]);
  int remove_weight = atoi(argv[4]);

  MPI_Init(&argc, &argv);
  // int nprocs;
  // int myrank;
  // MPI_Comm_size(MPI_COMM_WORLD, &nprocs);
  // MPI_Comm_rank(MPI_COMM_WORLD, &myrank);
  // char processor_name[MPI_MAX_PROCESSOR_NAME];
  // int name_len;
  // MPI_Get_processor_name(processor_name, &name_len);
  // printf("Hello world from processor %s, rank %d out of %d processors\n", processor_name, myrank, nprocs);

  ngu::HashMapWorkload workloadA(total_op, insert_weight, get_weight, remove_weight);

  ngu::benchmarkInsertedLFDHM(workloadA);

  ngu::benchmarkInsertedMHT(workloadA, 200);
  ngu::benchmarkInsertedMHT(workloadA, 400);
  ngu::benchmarkInsertedMHT(workloadA, 800);

  ngu::benchmarkInsertedSOHT(workloadA, 200);
  ngu::benchmarkInsertedSOHT(workloadA, 400);
  ngu::benchmarkInsertedSOHT(workloadA, 800);

  ngu::benchmarkInsertedBCHT(workloadA, 50);
  ngu::benchmarkInsertedBCHT(workloadA, 70);
  ngu::benchmarkInsertedBCHT(workloadA, 90);

  ngu::benchmarkInsertedBCHT(workloadA, 50);
  ngu::benchmarkInsertedBCHT(workloadA, 70);
  ngu::benchmarkInsertedBCHT(workloadA, 90);

  ngu::benchmarkInsertedPHHT(workloadA, 50);
  ngu::benchmarkInsertedPHHT(workloadA, 70);
  MPI_Finalize();
  return 0;
}