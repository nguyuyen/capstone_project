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
  ngu::HashMapWorkload workloadA(total_op, insert_weight, get_weight, remove_weight);

  ngu::benchmarkInsertedWFHM(workloadA);

  ngu::benchmarkInsertedMHT(workloadA, 200);
  ngu::benchmarkInsertedMHT(workloadA, 500);
  ngu::benchmarkInsertedMHT(workloadA, 1000);

  ngu::benchmarkInsertedBCHT(workloadA, 50);
  ngu::benchmarkInsertedBCHT(workloadA, 70);
  ngu::benchmarkInsertedBCHT(workloadA, 90);
  MPI_Finalize();
  return 0;
}