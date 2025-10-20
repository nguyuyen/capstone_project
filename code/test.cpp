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
  int type = atoi(argv[1]);
  int total_op = atoi(argv[2]);
  int insert_weight = atoi(argv[3]);
  int get_weight = atoi(argv[4]);
  int remove_weight = atoi(argv[5]);

  MPI_Init(&argc, &argv);

  ngu::HashMapWorkload workloadA(total_op, insert_weight, get_weight, remove_weight);

  switch (type) {
    case 1:
      ngu::benchmarkInsertedLFDHM(workloadA);
      break;

    case 2:
      ngu::benchmarkInsertedMHT(workloadA, 200);
      ngu::benchmarkInsertedMHT(workloadA, 400);
      ngu::benchmarkInsertedMHT(workloadA, 800);
      break;

    case 3:
      ngu::benchmarkInsertedSOHT(workloadA, 200);
      ngu::benchmarkInsertedSOHT(workloadA, 400);
      ngu::benchmarkInsertedSOHT(workloadA, 800);
      break;

    case 4:
      ngu::benchmarkInsertedBCHT(workloadA, 50);
      ngu::benchmarkInsertedBCHT(workloadA, 70);
      ngu::benchmarkInsertedBCHT(workloadA, 90);
      break;

    case 5:
      ngu::benchmarkInsertedPHHT(workloadA, 50);
      ngu::benchmarkInsertedPHHT(workloadA, 70);
      break;

    default:
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

      ngu::benchmarkInsertedPHHT(workloadA, 50);
      ngu::benchmarkInsertedPHHT(workloadA, 70);
      break;
  }
  MPI_Finalize();
  return 0;
}