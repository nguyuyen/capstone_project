#define MEM_CHECK

#include <mpi.h>
#include <unistd.h>

#include <chrono>
#include <fstream>
#include <iostream>
#include <random>

#include "hash_map.h"
#include "test_bench.h"

// #define AFRAID

int main(int argc, char** argv) {
  int type = atoi(argv[1]);
  int number_of_test = atoi(argv[2]);
  int total_op = atoi(argv[3]);
  int insert_weight = atoi(argv[4]);
  int get_weight = atoi(argv[5]);
  int remove_weight = atoi(argv[6]);

  if (number_of_test < 1) {
    number_of_test = 1;
  }
  if (number_of_test > 5) {
    number_of_test = 5;
  }

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  double time_in_ms;

  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  ngu::HashMapWorkload workloadA(total_op, insert_weight, get_weight, remove_weight);

  MPI_Barrier(MPI_COMM_WORLD);
  start = std::chrono::high_resolution_clock::now();

  for (int i = 0; i < number_of_test; ++i) {
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

      case 51:
        ngu::benchmarkInsertedPHHT(workloadA, 50);
        ngu::benchmarkInsertedPHHT(workloadA, 70);
        ngu::benchmarkInsertedPHHT(workloadA, 90);
        break;

      case 10:
        ngu::benchmarkInsertedLFDHM(workloadA);

        ngu::benchmarkInsertedMHT(workloadA, 200);
        ngu::benchmarkInsertedMHT(workloadA, 400);
        ngu::benchmarkInsertedMHT(workloadA, 800);

        ngu::benchmarkInsertedBCHT(workloadA, 50);
        ngu::benchmarkInsertedBCHT(workloadA, 70);
        ngu::benchmarkInsertedBCHT(workloadA, 90);

        ngu::benchmarkInsertedPHHT(workloadA, 50);
        ngu::benchmarkInsertedPHHT(workloadA, 70);
        break;

      case 104:
        ngu::benchmarkInsertedLFDHM(workloadA);

        ngu::benchmarkInsertedMHT(workloadA, 200);
        ngu::benchmarkInsertedMHT(workloadA, 400);
        ngu::benchmarkInsertedMHT(workloadA, 800);

        // ngu::benchmarkInsertedBCHT(workloadA, 50);
        // ngu::benchmarkInsertedBCHT(workloadA, 70);
        // ngu::benchmarkInsertedBCHT(workloadA, 90);

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
  }

  end = std::chrono::high_resolution_clock::now();
  MPI_Barrier(MPI_COMM_WORLD);

  time_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  double total_time;
  MPI_Reduce(&time_in_ms, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);

  if (rank == 0) {
    std::cout << "Total time: " << total_time / 1000000 << "s" << std::endl;
  }

  MPI_Finalize();
  return 0;
}