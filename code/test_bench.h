#ifndef DISTRIBUTED_DATA_STRUCTURE_TEST_BENCH_H
#define DISTRIBUTED_DATA_STRUCTURE_TEST_BENCH_H

#include <mpi.h>

#include <chrono>
#include <iostream>
#include <random>
#include <string>

#include "hash_map.h"

namespace ngu {
class HashMapWorkload;
class TestResult;

void print(TestResult test_result, bool brief = false);

void benchmarkWFHM(HashMapWorkload workload);
void benchmarkMHT(HashMapWorkload workload, int load_factor_in_percent);
void benchmarkBCHT(HashMapWorkload workload, int load_factor_in_percent);

void benchmarkInsertedWFHM(HashMapWorkload workload);
void benchmarkInsertedMHT(HashMapWorkload workload, int load_factor_in_percent);
void benchmarkInsertedBCHT(HashMapWorkload workload, int load_factor_in_percent);

class HashMapWorkload {
 public:
  HashMapWorkload(int total_operation, int insert_weight, int get_weight, int remove_weight)
      : total_operation(total_operation),
        insert_weight(insert_weight),
        get_weight(get_weight),
        remove_weight(remove_weight) {};

 public:
  int total_operation;
  int insert_weight;
  int get_weight;
  int remove_weight;
};
class TestResult {
 public:
  TestResult(std::string name,
             int load_factor_in_percent,
             int nproc,
             int workload,
             int key_range,
             int insert_weight,
             int get_weight,
             int remove_weight,
             int succeed_op,
             double throughput_in_ms) : name(name),
                                        load_factor_in_percent(load_factor_in_percent),
                                        nproc(nproc),
                                        workload(workload),
                                        key_range(key_range),
                                        insert_weight(insert_weight),
                                        get_weight(get_weight),
                                        remove_weight(remove_weight),
                                        succeed_op(succeed_op),
                                        throughput_in_ms(throughput_in_ms) {}
  std::string name;
  int load_factor_in_percent;
  int nproc;
  int workload;
  int key_range;
  int insert_weight;
  int get_weight;
  int remove_weight;
  int succeed_op;
  double throughput_in_ms;
};

void print(TestResult test_result, bool brief) {
  if (brief == true) {
    // std::cout << "   --- " << test_result.name << " ---" << std::endl;
    // if (test_result.load_factor_in_percent != 0) {
    //   std::cout << "Load factor: " << test_result.load_factor_in_percent / 100 << "." << test_result.load_factor_in_percent % 100 << std::endl;
    // }
    // std::cout << "   Throughput: " << test_result.throughput_in_ms << " op/ms" << std::endl;
    // std::cout << "----------------------" << std::endl;

    std::cout << test_result.name;
    if (test_result.load_factor_in_percent != 0) {
      std::cout << "/" << test_result.load_factor_in_percent / 100 << "." << test_result.load_factor_in_percent % 100;
    }
    std::cout << ": " << test_result.throughput_in_ms << " op/ms" << std::endl;
    return;
  }
  int op_weight_total = test_result.insert_weight + test_result.get_weight + test_result.remove_weight;
  std::cout << "   --- " << test_result.name << " ---" << std::endl;
  if (test_result.load_factor_in_percent != 0) {
    std::cout << "Load factor: " << test_result.load_factor_in_percent / 100 << "." << test_result.load_factor_in_percent % 100 << std::endl;
  }
  std::cout << "Number of process: " << test_result.nproc << std::endl;
  std::cout << "Workload: " << test_result.workload << " operator" << std::endl;
  std::cout << "Key range: " << "[" << 0 << "," << test_result.workload - 1 << "]" << std::endl;
  std::cout << "Insert weight: " << test_result.insert_weight << "/" << op_weight_total << std::endl;
  std::cout << "Get weight   : " << test_result.get_weight << "/" << op_weight_total << std::endl;
  std::cout << "Remove weight: " << test_result.remove_weight << "/" << op_weight_total << std::endl;
  std::cout << "   Succeed op: " << test_result.succeed_op << "/" << test_result.workload << std::endl;
  std::cout << "   Throughput: " << test_result.throughput_in_ms << " op/ms" << std::endl;
  std::cout << "----------------------" << std::endl;
}

void benchmarkWFHM(HashMapWorkload workload) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int key;
  int value;
  int op;
  int op_insert_hmm = workload.insert_weight;
  int op_get_hmm = workload.insert_weight + workload.get_weight;
  int op_weight_total = workload.insert_weight + workload.get_weight + workload.remove_weight;

  std::random_device rd_key;
  std::mt19937 gen_key(rd_key());
  std::uniform_int_distribution<> distr_key(0, workload.total_operation - 1);

  std::random_device rd_op;
  std::mt19937 gen_op(rd_op());
  std::uniform_int_distribution<> distr_op(0, op_weight_total - 1);

  int op_number_each = workload.total_operation / size;

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  double time_in_ms;

  int op_succeed_count = 0;

  WaitFreeHashMap<int, int> hash_map(MPI_COMM_WORLD);

  MPI_Barrier(MPI_COMM_WORLD);
  start = std::chrono::high_resolution_clock::now();

  // TESTING
  for (int i = 0; i < op_number_each; ++i) {
    key = distr_key(gen_key);
    op = distr_op(gen_op);
    if (op < op_insert_hmm) {
      if (hash_map.insert(key, rank) == true)
        ++op_succeed_count;
    } else if (op < op_get_hmm) {
      if (hash_map.get(key, value) == true)
        ++op_succeed_count;
    } else {
      if (hash_map.remove(key) == true)
        ++op_succeed_count;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = std::chrono::high_resolution_clock::now();

  time_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  int total_op;
  int total_op_succeed;
  double total_time;

  MPI_Reduce(&time_in_ms, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&op_number_each, &total_op, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&op_succeed_count, &total_op_succeed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  double throughput = total_op * 1.0 / (total_time);

  if (rank == 0) {
    std::cout << "   --- WFHM ---" << std::endl;
    std::cout << "Number of process: " << size << std::endl;
    std::cout << "Workload: " << total_op << " operator" << std::endl;
    std::cout << "Key range: " << "[" << 0 << "," << workload.total_operation - 1 << "]" << std::endl;
    std::cout << "Insert weight: " << workload.insert_weight << "/" << op_weight_total << std::endl;
    std::cout << "Get weight   : " << workload.get_weight << "/" << op_weight_total << std::endl;
    std::cout << "Remove weight: " << workload.remove_weight << "/" << op_weight_total << std::endl;
    std::cout << "   Succeed op: " << total_op_succeed << "/" << total_op << std::endl;
    std::cout << "   Throughput: " << throughput * 1000 << " op/ms" << std::endl;
    std::cout << "----------------------" << std::endl;
  }
}
void benchmarkMHT(HashMapWorkload workload, int load_factor_in_percent) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int key;
  int value;
  int op;
  int op_insert_hmm = workload.insert_weight;
  int op_get_hmm = workload.insert_weight + workload.get_weight;
  int op_weight_total = workload.insert_weight + workload.get_weight + workload.remove_weight;

  std::random_device rd_key;
  std::mt19937 gen_key(rd_key());
  std::uniform_int_distribution<> distr_key(0, workload.total_operation - 1);

  std::random_device rd_op;
  std::mt19937 gen_op(rd_op());
  std::uniform_int_distribution<> distr_op(0, op_weight_total - 1);

  int op_number_each = workload.total_operation / size;

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  double time_in_ms;

  int op_succeed_count = 0;

  int estimate_insert_op_each = op_number_each * workload.insert_weight / op_weight_total;
  int table_length_each = estimate_insert_op_each * 100 / load_factor_in_percent;

  MichealHashTable<int, int> hash_map(MPI_COMM_WORLD, table_length_each);

  MPI_Barrier(MPI_COMM_WORLD);
  start = std::chrono::high_resolution_clock::now();

  // TESTING
  for (int i = 0; i < op_number_each; ++i) {
    key = distr_key(gen_key);
    op = distr_op(gen_op);
    if (op < op_insert_hmm) {
      if (hash_map.insert(key, rank) == true)
        ++op_succeed_count;
    } else if (op < op_get_hmm) {
      if (hash_map.get(key, value) == true)
        ++op_succeed_count;
    } else {
      if (hash_map.remove(key) == true)
        ++op_succeed_count;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = std::chrono::high_resolution_clock::now();

  time_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  int total_op;
  int total_op_succeed;
  double total_time;

  MPI_Reduce(&time_in_ms, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&op_number_each, &total_op, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&op_succeed_count, &total_op_succeed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  double throughput = total_op * 1.0 / (total_time);

  if (rank == 0) {
    TestResult test_result("MHT", load_factor_in_percent, size, total_op, workload.total_operation, workload.insert_weight, workload.get_weight, workload.remove_weight, total_op_succeed, throughput * 1000);
    print(test_result, true);
    // std::cout << "   --- MHT ---" << std::endl;
    // std::cout << "Number of process: " << size << std::endl;
    // std::cout << "Load factor: " << load_factor_in_percent / 100 << "." << load_factor_in_percent % 100 << std::endl;
    // std::cout << "Workload: " << total_op << " operator" << std::endl;
    // std::cout << "Key range: " << "[" << 0 << "," << workload.total_operation - 1 << "]" << std::endl;
    // std::cout << "Insert weight: " << workload.insert_weight << "/" << op_weight_total << std::endl;
    // std::cout << "Get weight   : " << workload.get_weight << "/" << op_weight_total << std::endl;
    // std::cout << "Remove weight: " << workload.remove_weight << "/" << op_weight_total << std::endl;
    // std::cout << "   Succeed op: " << total_op_succeed << "/" << total_op << std::endl;
    // std::cout << "   Throughput: " << throughput * 1000 << " op/ms" << std::endl;
    // std::cout << "----------------------" << std::endl;
  }
}
void benchmarkBCHT(HashMapWorkload workload, int load_factor_in_percent) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int key;
  int value;
  int op;
  int op_insert_hmm = workload.insert_weight;
  int op_get_hmm = workload.insert_weight + workload.get_weight;
  int op_weight_total = workload.insert_weight + workload.get_weight + workload.remove_weight;

  std::random_device rd_key;
  std::mt19937 gen_key(rd_key());
  std::uniform_int_distribution<> distr_key(0, workload.total_operation - 1);

  std::random_device rd_op;
  std::mt19937 gen_op(rd_op());
  std::uniform_int_distribution<> distr_op(0, op_weight_total - 1);

  int op_number_each = workload.total_operation / size;

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  double time_in_ms;

  int op_succeed_count = 0;

  int estimate_insert_op_each = op_number_each * workload.insert_weight / op_weight_total;
  int table_length_each = (estimate_insert_op_each * 100 / load_factor_in_percent / 4);

  BucketizedCuckooHashTable<int, int> hash_map(MPI_COMM_WORLD, table_length_each);

  MPI_Barrier(MPI_COMM_WORLD);
  start = std::chrono::high_resolution_clock::now();

  // TESTING
  for (int i = 0; i < op_number_each; ++i) {
    key = distr_key(gen_key);
    op = distr_op(gen_op);
    if (op < op_insert_hmm) {
      if (hash_map.insert(key, rank) == true)
        ++op_succeed_count;
    } else if (op < op_get_hmm) {
      if (hash_map.get(key, value) == true)
        ++op_succeed_count;
    } else {
      if (hash_map.remove(key) == true)
        ++op_succeed_count;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = std::chrono::high_resolution_clock::now();

  time_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  int total_op;
  int total_op_succeed;
  double total_time;

  MPI_Reduce(&time_in_ms, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&op_number_each, &total_op, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&op_succeed_count, &total_op_succeed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  double throughput = total_op * 1.0 / (total_time);

  if (rank == 0) {
    std::cout << "   --- BCHT ---" << std::endl;
    std::cout << "Number of process: " << size << std::endl;
    std::cout << "Load factor: " << "0." << load_factor_in_percent << std::endl;
    std::cout << "Workload: " << total_op << " operator" << std::endl;
    std::cout << "Key range: " << "[" << 0 << "," << workload.total_operation - 1 << "]" << std::endl;
    std::cout << "Insert weight: " << workload.insert_weight << "/" << op_weight_total << std::endl;
    std::cout << "Get weight   : " << workload.get_weight << "/" << op_weight_total << std::endl;
    std::cout << "Remove weight: " << workload.remove_weight << "/" << op_weight_total << std::endl;
    std::cout << "   Succeed op: " << total_op_succeed << "/" << total_op << std::endl;
    std::cout << "   Throughput: " << throughput * 1000 << " op/ms" << std::endl;
    std::cout << "----------------------" << std::endl;
  }
}
void benchmarkInsertedWFHM(HashMapWorkload workload) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int key;
  int value;
  int op;
  int op_insert_hmm = workload.insert_weight;
  int op_get_hmm = workload.insert_weight + workload.get_weight;
  int op_weight_total = workload.insert_weight + workload.get_weight + workload.remove_weight;

  std::random_device rd_key;
  std::mt19937 gen_key(rd_key());
  std::uniform_int_distribution<> distr_key(0, workload.total_operation - 1);

  std::random_device rd_op;
  std::mt19937 gen_op(rd_op());
  std::uniform_int_distribution<> distr_op(0, op_weight_total - 1);

  int op_number_each = workload.total_operation / size;

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  double time_in_ms;

  int op_succeed_count = 0;

  ///
  int prev_insert = workload.total_operation / 2;
  int insert_each_need = prev_insert / size;

  WaitFreeHashMap<int, int> hash_map(MPI_COMM_WORLD);
  for (int i = 0; i < insert_each_need; ++i) {
    hash_map.insert(insert_each_need * rank + i, rank);
  }

  // MPI_Barrier(MPI_COMM_WORLD);
  // if (rank == 0) {
  //   std::cout << "Prev insert " << prev_insert << " done" << std::endl;
  // }
  MPI_Barrier(MPI_COMM_WORLD);

  start = std::chrono::high_resolution_clock::now();

  // TESTING
  for (int i = 0; i < op_number_each; ++i) {
    key = distr_key(gen_key);
    op = distr_op(gen_op);
    if (op < op_insert_hmm) {
      if (hash_map.insert(key, rank) == true)
        ++op_succeed_count;
    } else if (op < op_get_hmm) {
      if (hash_map.get(key, value) == true)
        ++op_succeed_count;
    } else {
      if (hash_map.remove(key) == true)
        ++op_succeed_count;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = std::chrono::high_resolution_clock::now();

  time_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  int total_op;
  int total_op_succeed;
  double total_time;

  MPI_Reduce(&time_in_ms, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&op_number_each, &total_op, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&op_succeed_count, &total_op_succeed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  double throughput = total_op * 1.0 / (total_time);

  if (rank == 0) {
    TestResult test_result("WFHM", 0, size, total_op, workload.total_operation, workload.insert_weight, workload.get_weight, workload.remove_weight, total_op_succeed, throughput * 1000);
    print(test_result, true);
    // std::cout << "   --- WFHM ---" << std::endl;
    // std::cout << "Number of process: " << size << std::endl;
    // std::cout << "Workload: " << total_op << " operator" << std::endl;
    // std::cout << "Key range: " << "[" << 0 << "," << workload.total_operation - 1 << "]" << std::endl;
    // std::cout << "Insert weight: " << workload.insert_weight << "/" << op_weight_total << std::endl;
    // std::cout << "Get weight   : " << workload.get_weight << "/" << op_weight_total << std::endl;
    // std::cout << "Remove weight: " << workload.remove_weight << "/" << op_weight_total << std::endl;
    // std::cout << "   Succeed op: " << total_op_succeed << "/" << total_op << std::endl;
    // std::cout << "   Throughput: " << throughput * 1000 << " op/ms" << std::endl;
    // std::cout << "----------------------" << std::endl;
  }
}
void benchmarkInsertedMHT(HashMapWorkload workload, int load_factor_in_percent) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int key;
  int value;
  int op;
  int op_insert_hmm = workload.insert_weight;
  int op_get_hmm = workload.insert_weight + workload.get_weight;
  int op_weight_total = workload.insert_weight + workload.get_weight + workload.remove_weight;

  std::random_device rd_key;
  std::mt19937 gen_key(rd_key());
  std::uniform_int_distribution<> distr_key(0, workload.total_operation - 1);

  std::random_device rd_op;
  std::mt19937 gen_op(rd_op());
  std::uniform_int_distribution<> distr_op(0, op_weight_total - 1);

  int op_number_each = workload.total_operation / size;

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  double time_in_ms;

  int op_succeed_count = 0;

  ///
  int prev_insert = workload.total_operation / 2;
  int table_length_each = (prev_insert / (load_factor_in_percent / 100)) / size;
  int insert_each_need = prev_insert / size;

  MichealHashTable<int, int> hash_map(MPI_COMM_WORLD, table_length_each);
  for (int i = 0; i < insert_each_need; ++i) {
    hash_map.insert(insert_each_need * rank + i, rank);
  }

  // MPI_Barrier(MPI_COMM_WORLD);
  // if (rank == 0) {
  //   std::cout << "Prev insert " << prev_insert << " done" << std::endl;
  // }
  MPI_Barrier(MPI_COMM_WORLD);

  start = std::chrono::high_resolution_clock::now();

  // TESTING
  for (int i = 0; i < op_number_each; ++i) {
    key = distr_key(gen_key);
    op = distr_op(gen_op);
    if (op < op_insert_hmm) {
      if (hash_map.insert(key, rank) == true)
        ++op_succeed_count;
    } else if (op < op_get_hmm) {
      if (hash_map.get(key, value) == true)
        ++op_succeed_count;
    } else {
      if (hash_map.remove(key) == true)
        ++op_succeed_count;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = std::chrono::high_resolution_clock::now();

  time_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  int total_op;
  int total_op_succeed;
  double total_time;

  MPI_Reduce(&time_in_ms, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&op_number_each, &total_op, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&op_succeed_count, &total_op_succeed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  double throughput = total_op * 1.0 / (total_time);

  if (rank == 0) {
    TestResult test_result("MHT", load_factor_in_percent, size, total_op, workload.total_operation, workload.insert_weight, workload.get_weight, workload.remove_weight, total_op_succeed, throughput * 1000);
    print(test_result, true);
    // std::cout << "   --- MHT ---" << std::endl;
    // std::cout << "Number of process: " << size << std::endl;
    // std::cout << "Load factor: " << load_factor_in_percent / 100 << "." << load_factor_in_percent % 100 << std::endl;
    // std::cout << "Workload: " << total_op << " operator" << std::endl;
    // std::cout << "Key range: " << "[" << 0 << "," << workload.total_operation - 1 << "]" << std::endl;
    // std::cout << "Insert weight: " << workload.insert_weight << "/" << op_weight_total << std::endl;
    // std::cout << "Get weight   : " << workload.get_weight << "/" << op_weight_total << std::endl;
    // std::cout << "Remove weight: " << workload.remove_weight << "/" << op_weight_total << std::endl;
    // std::cout << "   Succeed op: " << total_op_succeed << "/" << total_op << std::endl;
    // std::cout << "   Throughput: " << throughput * 1000 << " op/ms" << std::endl;
    // std::cout << "----------------------" << std::endl;
  }
}
void benchmarkInsertedBCHT(HashMapWorkload workload, int load_factor_in_percent) {
  int rank, size;
  MPI_Comm_rank(MPI_COMM_WORLD, &rank);
  MPI_Comm_size(MPI_COMM_WORLD, &size);

  int key;
  int value;
  int op;
  int op_insert_hmm = workload.insert_weight;
  int op_get_hmm = workload.insert_weight + workload.get_weight;
  int op_weight_total = workload.insert_weight + workload.get_weight + workload.remove_weight;

  std::random_device rd_key;
  std::mt19937 gen_key(rd_key());
  std::uniform_int_distribution<> distr_key(0, workload.total_operation - 1);

  std::random_device rd_op;
  std::mt19937 gen_op(rd_op());
  std::uniform_int_distribution<> distr_op(0, op_weight_total - 1);

  int op_number_each = workload.total_operation / size;

  std::chrono::time_point<std::chrono::high_resolution_clock> start, end;
  double time_in_ms;

  int op_succeed_count = 0;
  ///
  int prev_insert = workload.total_operation / 2;
  int table_length_each = ceil(1.0 * prev_insert * 100 / load_factor_in_percent / 4 / size);
  int insert_each_need = prev_insert / size;

  BucketizedCuckooHashTable<int, int> hash_map(MPI_COMM_WORLD, table_length_each);
  for (int i = 0; i < insert_each_need; ++i) {
    hash_map.insert(insert_each_need * rank + i, rank);
  }

  // MPI_Barrier(MPI_COMM_WORLD);
  // if (rank == 0) {
  //   std::cout << "Prev insert " << prev_insert << " done" << std::endl;
  // }
  MPI_Barrier(MPI_COMM_WORLD);

  start = std::chrono::high_resolution_clock::now();

  // TESTING
  for (int i = 0; i < op_number_each; ++i) {
    key = distr_key(gen_key);
    op = distr_op(gen_op);
    if (op < op_insert_hmm) {
      if (hash_map.insert(key, rank) == true)
        ++op_succeed_count;
    } else if (op < op_get_hmm) {
      if (hash_map.get(key, value) == true)
        ++op_succeed_count;
    } else {
      if (hash_map.remove(key) == true)
        ++op_succeed_count;
    }
  }

  MPI_Barrier(MPI_COMM_WORLD);
  end = std::chrono::high_resolution_clock::now();

  time_in_ms = std::chrono::duration_cast<std::chrono::microseconds>(end - start).count();

  int total_op;
  int total_op_succeed;
  double total_time;

  MPI_Reduce(&time_in_ms, &total_time, 1, MPI_DOUBLE, MPI_MAX, 0, MPI_COMM_WORLD);
  MPI_Reduce(&op_number_each, &total_op, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);
  MPI_Reduce(&op_succeed_count, &total_op_succeed, 1, MPI_INT, MPI_SUM, 0, MPI_COMM_WORLD);

  double throughput = total_op * 1.0 / (total_time);

  if (rank == 0) {
    TestResult test_result("BCHT", load_factor_in_percent, size, total_op, workload.total_operation, workload.insert_weight, workload.get_weight, workload.remove_weight, total_op_succeed, throughput * 1000);
    print(test_result, true);
    // std::cout << "   --- BCHT ---" << std::endl;
    // std::cout << "Number of process: " << size << std::endl;
    // std::cout << "Load factor: " << "0." << load_factor_in_percent << std::endl;
    // std::cout << "Workload: " << total_op << " operator" << std::endl;
    // std::cout << "Key range: " << "[" << 0 << "," << workload.total_operation - 1 << "]" << std::endl;
    // std::cout << "Insert weight: " << workload.insert_weight << "/" << op_weight_total << std::endl;
    // std::cout << "Get weight   : " << workload.get_weight << "/" << op_weight_total << std::endl;
    // std::cout << "Remove weight: " << workload.remove_weight << "/" << op_weight_total << std::endl;
    // std::cout << "   Succeed op: " << total_op_succeed << "/" << total_op << std::endl;
    // std::cout << "   Throughput: " << throughput * 1000 << " op/ms" << std::endl;
    // std::cout << "----------------------" << std::endl;
  }
}
}  // namespace ngu

#endif  // DISTRIBUTED_DATA_STRUCTURE_TEST_BENCH_H