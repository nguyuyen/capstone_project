#include <mpi.h>

#include <iostream>
#include <random>

#include "hash_map.h"

// #define AFRAID

int main(int argc, char** argv) {
  MPI_Init(&argc, &argv);
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  ngu::WaitFreeHashMap<int, int> wai_free_hash_map;

  int n = atoi(argv[1]);
  int k;
  int v;
  bool b;
  std::random_device rd;                            // obtain a random number from hardware
  std::mt19937 gen(rd());                           // seed the generator
  std::uniform_int_distribution<> distr(0, n);      // define the range
  std::uniform_int_distribution<> distr2(1, size);  // define the range
  int count = 0;
  int loop = n;
#ifdef AFRAID
  loop = std::min(n, 1000);
#endif

  std::cout << "[[" << rank << "]]" << ": inserting..." << std::endl;
  for (int i = 0; i < loop; ++i) {
    k = distr(gen);
    v = rank + 1;
    b = wai_free_hash_map.insert(k, v);
    if (b == false) {
      ++count;
    }
  }
  std::cout << "[[" << rank << "]]" << ": insert fail: " << count << "/" << loop << std::endl;
  std::cout << "[[" << rank << "]]" << ": updating..." << std::endl;
  int count1 = 0;
  for (int i = 0; i < loop; ++i) {
    k = distr(gen);
    v = distr2(gen);
    b = wai_free_hash_map.update(k, v, rank + 1);
    if (b == false)
      ++count1;
  }
  std::cout << "[[" << rank << "]]" << ": update fail: " << count1 << "/" << loop << std::endl;
  int counter = 0;
  std::cout << "[[" << rank << "]]" << ": removeing..." << std::endl;
  for (int i = 0; i < loop; ++i) {
    k = distr(gen);
    v = rank + 1;
    b = wai_free_hash_map.remove(k, v);
    if (b == false) {
      ++counter;
    }
  }
  std::cout << "[[" << rank << "]]" << ": remove fail: " << count << "/" << loop << std::endl;
  int count2 = 0;
  std::cout << "[[" << rank << "]]" << ": geting..." << std::endl;
  for (int i = 0; i < loop; ++i) {
    k = distr(gen);
    v = wai_free_hash_map.get(k);
    if (v == 0)
      ++count2;
  }
  std::cout << "[[" << rank << "]]" << ": get fail: " << count2 << "/" << loop << std::endl;
  MPI_Finalize();
  return 0;
}