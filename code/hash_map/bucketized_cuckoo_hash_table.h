#ifndef DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_BUCKETIZED_CUCKOO_HASHTABLE_H
#define DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_BUCKETIZED_CUCKOO_HASHTABLE_H

// #define COMM_CHECK

#ifdef DEBUG
#include <unistd.h>
#endif  // DEBUG

#ifdef DEBUG2
#include <unistd.h>

#include <map>
#endif  // DEBUG

#ifdef DEBUG_MOVE
#include <map>
#endif  // DEBUG_MOVE

#include <mpi.h>

#include <algorithm>
#include <array>
#include <bitset>
#include <cmath>
#include <iostream>
#include <list>
#include <queue>
#include <random>
#include <set>

#include "../lib/hash_function.h"

namespace ngu {
uint64_t splitmix64(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x = x ^ (x >> 31);
  return x;
}

template <typename Key, int Seed = 2510>
class Hash3 {
 public:
  uint64_t operator()(Key key) const {
    std::hash<Key> h;
    return splitmix64(h(key) + Seed);
  }
};

template <typename Key, typename Value, typename HashFunctor1 = Hash3<Key, 833123>, typename HashFunctor2 = Hash3<Key, 33491>, typename HashFunctor3 = Hash3<Key, 895>>
class BucketizedCuckooHashTable {
  /// @brief rank(11 bit) - disp(48 bit) - tag(5 bit) [unuse(2bit)-target(2bit)-kick(1bit)]
  using GPtr = uint64_t;
  class DataNode;
  class SlotInfo;
  class CuckooRecord;
  class Snapshot;

 private:
  static const GPtr null_gptr = 0;
  static constexpr int max_depth_search = 5;

 public:
  BucketizedCuckooHashTable(MPI_Comm comm = MPI_COMM_WORLD, int table_length_each = 1000);
  BucketizedCuckooHashTable(const BucketizedCuckooHashTable&) = delete;
  BucketizedCuckooHashTable& operator=(const BucketizedCuckooHashTable&) = delete;
  ~BucketizedCuckooHashTable();

  bool insert(Key key, Value value);
  bool get(Key key, Value& value);
  bool remove(Key key);

  // #ifdef MEM_CHECK
  int getMem() {
    return (sizeof(GPtr) * this->table_length_each +  // table
            sizeof(DataNode) * data_mem.size());      // data
  }
  // #endif  // MEM_CHECK

#ifdef COMM_CHECK
  int countIntra() {
    return this->intra_comm_count;
  }
  int countInter() {
    return this->inter_comm_count;
  }
#endif  // COMM_CHECK

 private:
  /// @brief Search key in 2 bucket
  /// @param key Key
  /// @param bucket Bucket hit if hit, bucket miss if has empty slot
  /// @param slot Slot hit if hit, slot miss if has empty slot
  /// @return 0: hit node has same key, 1 empty slot, 2 full of slot
  int search(Key key, int& bucket, int& slot, Value& value);
  /// @brief Move item from source to target
  /// @param source_bucket Source bucket
  /// @param source_slot Source slot
  /// @param target_bucket Target bucket
  /// @param target_slot Target slot
  /// @return True if succeed, otherwise false.
  bool moveItem(int source_bucket, int source_slot, int target_bucket, int target_slot);
  /// @brief Help move item if source node isKickMarked.
  /// @param source_bucket Source bucket
  /// @param source_slot Source slot
  /// @param source_node Source node
  void helper(int source_bucket, int source_slot, GPtr source_node);
  /// @brief Copy kick marked source node to target, them remove item at source.
  /// @param source_bucket
  /// @param source_slot
  /// @param target_bucket
  /// @param target_slot
  /// @param kick_marked_node
  /// @return True if succeed, otherwise false.
  bool copy(int source_bucket, int source_slot, int target_bucket, int target_slot, GPtr kick_marked_node);
  /// @brief Notify searching process that searched item may moved.
  /// @param hash_value
  void setRetryIfHazard(uint64_t hash_value);
  /// @brief Search nearest empty slot from 2 bucket.
  /// @param bucket1
  /// @param bucket2
  /// @return Path code to move if found.
  SlotInfo slotSearch(int bucket1, int bucket2);
  /// @brief Find path to get empty slot
  /// @param bucket1
  /// @param bucket2
  /// @param path Output path.
  /// @return Depth of path.
  int pathSearch(int bucket1, int bucket2, std::array<CuckooRecord, max_depth_search>& path);
  /// @brief Move item along path to get empty slot
  /// @param path
  /// @param depth
  /// @return True if succeed, otherwise false.
  bool moveAlongPath(std::array<CuckooRecord, max_depth_search>& path, int depth);
  /// @brief Check if key is duplicate, if yes, remove it
  /// @param key
  void checkDuplicateKey(Key key);

 private:
  GPtr getNode(int bucket, int slot);
  void getBucketData(int bucket, GPtr* arr);
  DataNode getData(GPtr node);
  void setData(GPtr node, Key key, Value value);
  GPtr allocateNode(Key key, Value value);
  void safeFreeNode(GPtr node);
  void freeNode(GPtr node);
  void scan();

 private:
  GPtr makeGPtr(int rank, MPI_Aint disp, int tag);
  int getRankt(GPtr node);
  MPI_Aint getDispt(GPtr node);
  int getRankByBucket(int bucket);
  MPI_Aint getDispBySlot(int bucket, int slot);
  bool CAS(int bucket, int slot, GPtr old_value, GPtr new_value);
  GPtr kickMark(GPtr node, int target_slot);
  bool isKickMarked(GPtr node);
  GPtr unkickMarked(GPtr node);
  int getTargetSlot(GPtr node);
  void calBucket(Key key, int& bucket1, int& bucket2);
  void calHash(Key key, uint64_t& hash_value1, uint64_t& hash_value2);
  void calHashAndBucket(Key key, uint64_t& hash_value1, uint64_t& hash_value2, int& bucket1, int& bucket2);
  void calBucketFromHash(int& bucket1, int& bucket2, uint64_t& hash_value1, uint64_t& hash_value2);
  uint64_t getHpRecord(int rank);
  void setHpRecord(uint64_t hash_value);
  uint64_t getHpFlag(int rank);
  void setHpFlag(int rank, uint64_t hash_value);
  GPtr getHPtr(int rank, int number);
  void setHPtr(GPtr node, int number);

 private:
  int nprocs;
  int myrank;
  int table_length_each;
  HashFunctor1 hash_functor1;
  HashFunctor2 hash_functor2;
  HashFunctor3 hash_functor_backup;
  std::size_t max_dlist_size = 10;
  int max_insert_try = 10;
#ifdef DEBUG_MOVE
  int move_count = 0;
#endif  // DEBUG_MOVE
#ifdef DEBUG_DUPLICATE
  int duplicate_count = 0;
#endif  // DEBUG_DUPLICATE
#ifdef COMM_CHECK
  bool isIntra(int source_rank, int target_rank) {
    return (source_rank / 8) == (target_rank / 8);
  }
  int intra_comm_count = 0;
  int inter_comm_count = 0;
#endif  // COMM_CHECK

 private:
  MPI_Comm comm;
  MPI_Win table;
  MPI_Win data;

 private:
  std::list<DataNode*> data_mem;
  MPI_Win hp_flag_win, hp_record_win;
  uint64_t *hp_flag, *hp_record;
  MPI_Win hp_win;
  GPtr* hp;
  std::list<GPtr> plist;
  std::list<GPtr> rlist;
  std::list<GPtr> dlist;
  GPtr* table_arr;

  class DataNode {
   public:
    DataNode() : key(), value() {};
    DataNode(Key key, Value value) : key(key), value(value) {};

   public:
    Key key;
    Value value;
  };
  class SlotInfo {
   public:
    SlotInfo() {}
    SlotInfo(int bucket, uint16_t path_code, int8_t depth) : bucket(bucket), path_code(path_code), depth(depth) {}

   public:
    int bucket;
    uint16_t path_code;
    int8_t depth;
  };
  /// @brief Save info about path to kick: bucket, slot, hash_value1, hash_value2
  class CuckooRecord {
   public:
    int bucket;
    int slot;
    uint64_t hash_value1;
    uint64_t hash_value2;
  };
  class Snapshot {
   public:
    Snapshot() {}
    Snapshot(int bucket, int slot, GPtr node) : bucket(bucket), slot(slot), node(node) {}

   public:
    int bucket;
    int slot;
    GPtr node;
  };
};

template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::BucketizedCuckooHashTable(MPI_Comm comm, int table_length_each)
    : table_length_each(table_length_each), comm(comm), data_mem(), plist(), rlist(), dlist() {
  MPI_Comm_size(comm, &this->nprocs);
  MPI_Comm_rank(comm, &this->myrank);

  MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &this->data);

  this->table_arr = new GPtr[this->table_length_each * 4]();
  MPI_Win_create(this->table_arr, (this->table_length_each * 4 * sizeof(GPtr)), sizeof(GPtr), MPI_INFO_NULL, comm, &this->table);

  this->hp_record = new uint64_t();
  MPI_Win_create(this->hp_record, sizeof(uint64_t), sizeof(uint64_t), MPI_INFO_NULL, comm, &this->hp_record_win);

  this->hp_flag = new uint64_t();
  MPI_Win_create(this->hp_flag, sizeof(uint64_t), sizeof(uint64_t), MPI_INFO_NULL, comm, &this->hp_flag_win);

  this->hp = new GPtr[2]();
  MPI_Win_create(this->hp, sizeof(GPtr) * 2, sizeof(GPtr), MPI_INFO_NULL, comm, &this->hp_win);
#ifdef DEBUG_HASH
  if (this->myrank == 0) {
    int temp_bucket1, temp_bucket2;
    int number_key_try = (this->table_length_each * this->nprocs) * 10;
    int number_of_bucket = this->table_length_each * this->nprocs;
    int* count_bucket_hit = new int[number_of_bucket]();
    for (int i = 0; i < number_key_try; ++i) {
      this->calBucket(i, temp_bucket1, temp_bucket2);
      ++(count_bucket_hit[temp_bucket1]);
      ++(count_bucket_hit[temp_bucket2]);
    }
    int min_bucket_hit = number_key_try;
    int max_bucket_hit = 0;
    double mean_bucket_hit = 0;
    double M2 = 0;
    int welford_count = 0;

    for (int i = 0; i < number_of_bucket; ++i) {
      if (min_bucket_hit > count_bucket_hit[i])
        min_bucket_hit = count_bucket_hit[i];
      if (max_bucket_hit < count_bucket_hit[i])
        max_bucket_hit = count_bucket_hit[i];

      ++welford_count;
      double x = count_bucket_hit[i];
      double delta = x - mean_bucket_hit;
      mean_bucket_hit += delta / welford_count;
      double delta2 = x - mean_bucket_hit;
      M2 += delta * delta2;
    }
    double var_bucket_hit = 0;
    if (welford_count > 0) {
      var_bucket_hit = M2 / welford_count;
    }

    std::cout << "Number of buckets = " << number_of_bucket << std::endl;
    std::cout << "Sample = " << number_key_try << std::endl;
    std::cout << "Min  = " << min_bucket_hit << std::endl;
    std::cout << "Max  = " << max_bucket_hit << std::endl;
    std::cout << "Mean = " << mean_bucket_hit << std::endl;
    std::cout << "STD  = " << sqrt(var_bucket_hit) << std::endl;

    delete[] count_bucket_hit;
  }
  MPI_Barrier(comm);
#endif  // DEBUG_HASH

  MPI_Win_fence(0, this->table);
  MPI_Win_fence(0, this->data);
  MPI_Win_fence(0, this->hp_record_win);
  MPI_Win_fence(0, this->hp_flag_win);
  MPI_Win_fence(0, this->hp_win);
  MPI_Win_lock_all(0, this->table);
  MPI_Win_lock_all(0, this->data);
  MPI_Win_lock_all(0, this->hp_record_win);
  MPI_Win_lock_all(0, this->hp_flag_win);
  MPI_Win_lock_all(0, this->hp_win);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::~BucketizedCuckooHashTable() {
#ifdef DEBUG_MOVE
  std::cout << "[[Rank = " << this->myrank << "]]: " << "move count = " << this->move_count << std::endl;
#endif  // DEBUG_MOVE
#ifdef DEBUG_DUPLICATE
  std::cout << "[[Rank = " << this->myrank << "]]: " << "duplicate count = " << this->duplicate_count << std::endl;
#endif  // DEBUG_DUPLICATE
  int flag;
  delete[] this->table_arr;
  for (typename std::list<DataNode*>::iterator it = data_mem.begin(); it != data_mem.end(); ++it) {
    delete *it;
  }
  delete this->hp_flag;
  delete this->hp_record;
  delete[] this->hp;
  MPI_Finalized(&flag);
  if (!flag) {
    MPI_Win_unlock_all(this->table);
    MPI_Win_unlock_all(this->data);
    MPI_Win_unlock_all(this->hp_record_win);
    MPI_Win_unlock_all(this->hp_flag_win);
    MPI_Win_unlock_all(this->hp_win);
    MPI_Win_free(&this->table);
    MPI_Win_free(&this->data);
    MPI_Win_free(&this->hp_record_win);
    MPI_Win_free(&this->hp_flag_win);
    MPI_Win_free(&this->hp_win);
  }
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline bool BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::insert(Key key, Value value) {
#ifdef DEBUG
  std::cout << "[[Rank = " << this->myrank << "]]: " << "insert <" << key << "," << value << ">" << std::endl;
#endif  // DEBUG
  int bucket1, bucket2;
  this->calBucket(key, bucket1, bucket2);
  int bucket, slot;
  int search_result;
  Value value_temp;
  bool result = false;
  GPtr node = this->allocateNode(key, value);
  int insert_try_count = 0;
#ifdef DEBUG2
  int debug_count = 0;
  int debug_max_count = 10;
#endif  // DEBUG2
  while (true) {
    ++insert_try_count;
    if (insert_try_count > this->max_insert_try)
      break;
#ifdef DEBUG2
    ++debug_count;
    if (debug_count >= debug_max_count) {
      if (debug_count == debug_max_count) {
        std::cout << "[Rank = " << this->myrank << "]: " << "insert fail loop 1." << std::endl;
      }
      while (true) {
        sleep(5);
      }
    }
#endif  // DEBUG2
    search_result = this->search(key, bucket, slot, value_temp);
    if (search_result == 0) {
      result = false;
      break;
    } else if (search_result == 1) {
      if (this->CAS(bucket, slot, null_gptr, node) == true) {
        result = true;
        break;
      } else
        continue;
    } else {
      std::array<CuckooRecord, max_depth_search> path;
      int depth = this->pathSearch(bucket1, bucket2, path);
      this->moveAlongPath(path, depth);
      continue;
    }
  }
  if (result == false) {
    this->freeNode(node);
  } else {
    this->checkDuplicateKey(key);
  }
#ifdef DEBUG
  std::cout << "[[Rank = " << this->myrank << "]]: " << "insert <" << key << "," << value << ">" << ((result == true) ? "succeed" : "fail") << std::endl;
#else
#ifdef DEBUG2
  if (result == true)
    std::cout << "[[Rank = " << this->myrank << "]]: " << "inserted <" << key << "> at bucket: " << bucket << std::endl;
#endif  // DEBUG2
#endif  // DEBUG
  return result;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline bool BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::get(Key key, Value& value) {
#ifdef DEBUG
  std::cout << "[[Rank = " << this->myrank << "]]: " << "get key: " << key << std::endl;
#endif  // DEBUG
  int bucket, slot;
  if (this->search(key, bucket, slot, value) == 0) {
    return true;
  } else
    return false;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline bool BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::remove(Key key) {
#ifdef DEBUG
  std::cout << "[[Rank = " << this->myrank << "]]: " << "remove key: " << key << std::endl;
#endif  // DEBUG
  int bucket, slot;
  Value value_temp;
  bool result = false;
  while (true) {
    if (this->search(key, bucket, slot, value_temp) == 0) {
      GPtr node = getNode(bucket, slot);
      this->setHPtr(node, 0);
      if (this->getNode(bucket, slot) != node)
        continue;
      if (this->CAS(bucket, slot, node, null_gptr) == true) {
        this->safeFreeNode(node);
        result = true;
        break;
      } else
        continue;
    } else {
      result = false;
      break;
    }
  }
  this->setHPtr(null_gptr, 0);
  return result;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline int BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::search(Key key, int& bucket, int& slot, Value& value) {
  bool reset;
  bool hit = false;
  bool miss = false;
  int bucket1, bucket2;
  uint64_t hash_value1, hash_value2;
  this->calHashAndBucket(key, hash_value1, hash_value2, bucket1, bucket2);
#ifdef DEBUG
  std::cout << "[[Rank = " << this->myrank << "]]: " << "search at bucket: " << bucket1 << "," << bucket2 << std::endl;
#endif  // DEBUG
  this->setHpRecord(hash_value1);
  GPtr bucket_data[8];
#ifdef DEBUG2
  int debug_count = 0;
  int debug_max_count = 10;
#endif  // DEBUG2
  while (true) {
#ifdef DEBUG2
    ++debug_count;
    if (debug_count >= debug_max_count) {
      if (debug_count == debug_max_count) {
        std::cout << "[Rank = " << this->myrank << "]: " << "search fail loop 1." << std::endl;
        std::cout << "[Rank = " << this->myrank << "]:";
        for (int i = 0; i < 8; ++i) {
          std::cout << " " << bucket_data[i];
        }
        std::cout << std::endl;
      }
      while (true) {
        sleep(5);
      }
    }
#endif  // DEBUG2
    reset = false;
    hit = false;
    miss = false;
    this->getBucketData(bucket1, bucket_data);
    this->getBucketData(bucket2, bucket_data + 4);
    for (int i = 0; i < 8; ++i) {
      this->setHPtr(bucket_data[i], 0);
      if (this->getNode(((i < 4) ? bucket1 : bucket2), i % 4) != bucket_data[i]) {
        reset = true;
        break;
      }
      if (bucket_data[i] == null_gptr) {
        if (miss == false) {
          miss = true;
          bucket = ((i < 4) ? bucket1 : bucket2);
          slot = i % 4;
        }
      } else {
        if (this->isKickMarked(bucket_data[i]) == true) {
          this->helper(((i < 4) ? bucket1 : bucket2), i % 4, bucket_data[i]);
          reset = true;
          break;
        }
        DataNode data_node = this->getData(bucket_data[i]);
        value = data_node.value;
        if (data_node.key == key) {
          hit = true;
          bucket = ((i < 4) ? bucket1 : bucket2);
          slot = i % 4;
          break;
        }
      }
    }
    this->setHPtr(null_gptr, 0);
    if (reset == true) {
      continue;
    }
    if (hit == true) {
#ifdef DEBUG
      std::cout << "[[Rank = " << this->myrank << "]]: " << "search at bucket: " << bucket1 << "," << bucket2 << ": found" << std::endl;
#endif  // DEBUG
      return 0;
    } else if (this->getHpFlag(this->myrank) == 1) {
      this->setHpFlag(this->myrank, 0);
      continue;
    } else if (miss == true) {
#ifdef DEBUG
      std::cout << "[[Rank = " << this->myrank << "]]: " << "search at bucket: " << bucket1 << "," << bucket2 << ": has slot" << std::endl;
      std::cout << "[Rank = " << this->myrank << "]:";
      for (int i = 0; i < 8; ++i) {
        std::cout << " " << bucket_data[i];
      }
      std::cout << std::endl;
#endif  // DEBUG
      return 1;
    } else {
#ifdef DEBUG
      std::cout << "[[Rank = " << this->myrank << "]]: " << "search at bucket: " << bucket1 << "," << bucket2 << ": full" << std::endl;
#else
#ifdef DEBUG2
      std::cout << "[[Rank = " << this->myrank << "]]: " << "search at bucket: " << bucket1 << "," << bucket2 << ": full" << std::endl;
#endif  // DEBUG2
#endif  // DEBUG
      return 2;
    }
  }
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline bool BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::moveItem(int source_bucket, int source_slot, int target_bucket, int target_slot) {
#ifdef DEBUG2
  std::cout << "[[Rank = " << this->myrank << "]]: " << " move: <" << source_bucket << "," << source_slot << ">-><" << target_bucket << "," << target_slot << ">" << std::endl;
#endif  // DEBUG2
  while (true) {
    GPtr source_node = this->getNode(source_bucket, source_slot);
    this->setHPtr(source_node, 0);
    if (source_node != this->getNode(source_bucket, source_slot))
      continue;
    if (source_node == null_gptr)
      return true;
    if (this->isKickMarked(source_node) == true) {
      this->helper(source_bucket, source_slot, source_node);
      this->setHPtr(null_gptr, 0);
      return false;
    }
    GPtr kick_marked_source_node = this->kickMark(source_node, target_slot);
    if (this->CAS(source_bucket, source_slot, source_node, kick_marked_source_node) == false) {
      this->setHPtr(null_gptr, 0);
      return false;
    }
    return this->copy(source_bucket, source_slot, target_bucket, target_slot, kick_marked_source_node);
  }
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::helper(int source_bucket, int source_slot, GPtr source_node) {
  Key key = this->getData(source_node).key;
  int target_bucket1, target_bucket2;
  this->calBucket(key, target_bucket1, target_bucket2);
  int target_bucket = ((target_bucket1 == source_bucket) ? target_bucket2 : target_bucket1);
  int target_slot = this->getTargetSlot(source_node);
  this->copy(source_bucket, source_slot, target_bucket, target_slot, source_node);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline bool BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::copy(int source_bucket, int source_slot, int target_bucket, int target_slot, GPtr kick_marked_node) {
  GPtr source_node = this->unkickMarked(kick_marked_node);
  Key key;
  uint64_t hash_value1, hash_value2;
  if (this->CAS(target_bucket, target_slot, null_gptr, source_node) == true) {
    key = this->getData(source_node).key;
    this->calHash(key, hash_value1, hash_value2);
    this->setRetryIfHazard(hash_value1);
    if (this->CAS(source_bucket, source_slot, kick_marked_node, null_gptr) == true)
      return true;
  }
  // TODO
  GPtr target_node;
  while (true) {
    target_node = this->getNode(target_bucket, target_slot);
    this->setHPtr(target_node, 1);
    if (this->getNode(target_bucket, target_slot) == target_node)
      break;
  }

  key = this->getData(source_node).key;
  if (key == this->getData(target_node).key) {
    this->calHash(key, hash_value1, hash_value2);
    this->setRetryIfHazard(hash_value1);
    this->CAS(source_bucket, source_slot, kick_marked_node, null_gptr);
    this->setHPtr(null_gptr, 0);
    this->setHPtr(null_gptr, 1);
    return false;
  }
  this->CAS(source_bucket, source_slot, kick_marked_node, null_gptr);
  this->setHPtr(null_gptr, 0);
  this->setHPtr(null_gptr, 1);
  return false;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::setRetryIfHazard(uint64_t hash_value) {
  for (int i = 0; i < this->nprocs; ++i) {
    if (hash_value == this->getHpRecord(i))
      this->setHpFlag(i, 1);
  }
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline typename BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::SlotInfo
BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::slotSearch(int bucket1, int bucket2) {
#ifdef DEBUG2
  std::cout << "[[Rank = " << this->myrank << "]]: " << " slot search at bucket: " << bucket1 << "," << bucket2 << std::endl;
#endif  // DEBUG2
  std::queue<SlotInfo> q;
  q.push(SlotInfo(bucket1, 0, 0));
  q.push(SlotInfo(bucket2, 1, 0));
  std::set<int> s;
  s.insert(bucket1);
  s.insert(bucket2);
  GPtr bucket_data[4];
  int slot = 0;
  bool reset;
  while (q.empty() == false) {
    reset = false;
    SlotInfo x = q.front();
    this->getBucketData(x.bucket, bucket_data);
#ifdef DEBUG2
    std::cout << "[[Rank = " << this->myrank << "]]: " << "search depth: " << (int)(x.depth) << ", bucket: " << x.bucket << " ";
    for (int i = 0; i < 4; ++i) {
      std::cout << "|" << bucket_data[i];
    }
    std::cout << "|" << std::endl;
#endif  // DEBUG2
    for (slot = 0; slot < 4; ++slot) {
      this->setHPtr(bucket_data[slot], 0);
      if (this->getNode(x.bucket, slot) != bucket_data[slot]) {
        reset = true;
        break;
      }
      if (bucket_data[slot] == null_gptr) {
        x.path_code = x.path_code * 4 + slot;
#ifdef DEBUG2
        std::cout << "   [[Rank = " << this->myrank << "]]: " << " slot search succeed at depth: " << (int)(x.depth) << std::endl;
#endif  // DEBUG2
        return x;
      }
      if (this->isKickMarked(bucket_data[slot]) == true) {
        this->helper(x.bucket, slot, bucket_data[slot]);
        reset = true;
        break;
      }
      if (x.depth < this->max_depth_search - 1) {
        Key key = this->getData(bucket_data[slot]).key;
        int bucket_temp1, bucket_temp2;
        this->calBucket(key, bucket_temp1, bucket_temp2);
        int bucket = ((x.bucket == bucket_temp1) ? bucket_temp2 : bucket_temp1);
        if (s.find(bucket) == s.end()) {
          s.insert(bucket);
          q.push(SlotInfo(bucket, x.path_code * 4 + slot, x.depth + 1));
        }
      }
    }
    this->setHPtr(null_gptr, 0);
    if (reset == true) {
      continue;
    } else {
      q.pop();
    }
  }
#ifdef DEBUG2
  std::cout << "[[Rank = " << this->myrank << "]]: " << " slot search fail" << std::endl;
#endif  // DEBUG2
  return SlotInfo(0, 0, -1);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline int BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::pathSearch(int bucket1, int bucket2, std::array<CuckooRecord, max_depth_search>& path) {
#ifdef DEBUG2
  std::cout << "[[Rank = " << this->myrank << "]]: " << " path search at bucket: " << bucket1 << "," << bucket2 << std::endl;
#endif  // DEBUG2
  bool reset;
  while (true) {
    reset = false;
    SlotInfo x = this->slotSearch(bucket1, bucket2);
    if (x.depth == -1)
      return -1;
    for (int i = x.depth; i >= 0; --i) {
      path[i].slot = x.path_code % 4;
      x.path_code /= 4;
    }
    CuckooRecord& first = path[0];
    if (x.path_code == 0)
      first.bucket = bucket1;
    else
      first.bucket = bucket2;
    GPtr node = this->getNode(first.bucket, first.slot);
    this->setHPtr(node, 0);
    if (this->getNode(first.bucket, first.slot) != node)
      continue;
    if (node == null_gptr)
      return 0;
    if (this->isKickMarked(node) == true) {
      this->helper(first.bucket, first.slot, node);
      continue;
    }
    calHash(this->getData(node).key, first.hash_value1, first.hash_value2);

    for (int i = 1; i <= x.depth; ++i) {
      CuckooRecord& curr = path[i];
      CuckooRecord& prev = path[i - 1];
      int bucket_temp1, bucket_temp2;
      this->calBucketFromHash(bucket_temp1, bucket_temp2, prev.hash_value1, prev.hash_value2);
      curr.bucket = ((prev.bucket == bucket_temp1) ? bucket_temp2 : bucket_temp1);
      node = this->getNode(curr.bucket, curr.slot);
      this->setHPtr(node, 0);
      if (this->getNode(curr.bucket, curr.slot) != node) {
        reset = true;
        break;
      }
      if (node == null_gptr)
        return i;
      if (this->isKickMarked(node) == true) {
        this->helper(curr.bucket, curr.slot, node);
        reset = true;
        break;
      }
      calHash(this->getData(node).key, curr.hash_value1, curr.hash_value2);
    }
    if (reset == true)
      continue;
    this->setHPtr(null_gptr, 0);
    return x.depth;
  }
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline bool BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::moveAlongPath(std::array<CuckooRecord, max_depth_search>& path, int depth) {
#ifdef DEBUG_MOVE
#ifdef DEBUG2
  std::cout << "[[Rank = " << this->myrank << "]]: " << " move along path, depth: " << depth << std::endl;
#endif  // DEBUG2
  this->move_count++;
#endif  // DEBUG_MOVE
  if (depth == 0) {
    if (getNode(path[0].bucket, path[0].slot) == null_gptr)
      return true;
    else
      return false;
  }
  while (depth > 0) {
    if (this->moveItem(path[depth - 1].bucket, path[depth - 1].slot, path[depth].bucket, path[depth].slot) == false)
      return false;
    --depth;
  }
  return true;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::checkDuplicateKey(Key key) {
  bool reset = false;
  int count = 0;
  Snapshot task;
  int bucket1, bucket2;
  calBucket(key, bucket1, bucket2);
  while (true) {
    reset = false;
    count = 0;
    std::list<Snapshot> l;
    GPtr bucket_data[8];
    this->getBucketData(bucket1, bucket_data);
    this->getBucketData(bucket2, bucket_data + 4);
    for (int i = 0; i < 8; ++i) {
      this->setHPtr(bucket_data[i], 0);
      if (this->getNode(((i < 4) ? bucket1 : bucket2), i % 4) != bucket_data[i]) {
        reset = true;
        break;
      }
      if (bucket_data[i] != null_gptr) {
        l.push_back(Snapshot(((i < 4) ? bucket1 : bucket2), i % 4, bucket_data[i]));
      }
    }
    if (this->getHpFlag(this->myrank) == 1) {
      this->setHpFlag(this->myrank, 0);
      reset = true;
      continue;
    }
    while (l.empty() == false) {
      Snapshot s = l.back();
      this->setHPtr(s.node, 0);
      if (this->getNode(s.bucket, s.slot) != s.node) {
        reset = true;
        break;
      }
      l.pop_back();
      if (this->isKickMarked(s.node)) {
        this->helper(s.bucket, s.slot, s.node);
        reset = true;
        break;
      }
      Key key_temp = this->getData(s.node).key;
      if (key_temp == key) {
        task = Snapshot(s.bucket, s.slot, s.node);
        ++count;
      }
    }
    if (reset == true)
      continue;
    if (count <= 1) {
      this->setHPtr(null_gptr, 0);
      return;
    } else {
#ifdef DEBUG_DUPLICATE
      ++(this->duplicate_count);
#endif  // DEBUG_DUPLICATE
#ifdef DEBUG2
      std::cout << "[" << this->myrank << "]: " << "has duplicate key: " << key << std::endl;
      bool cas_result = this->CAS(task.bucket, task.slot, task.node, null_gptr);
      if (cas_result == true) {
        std::cout << "[" << this->myrank << "]: " << "remove duplicate succeed." << std::endl;
      }
#else
      this->CAS(task.bucket, task.slot, task.node, null_gptr);
#endif  // DEBUG2
      continue;
    }
  }
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline typename BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::GPtr
BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::getNode(int bucket, int slot) {
  GPtr result = 2;
  int rank = this->getRankByBucket(bucket);
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  int disp = this->getDispBySlot(bucket, slot);
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(GPtr), MPI_CHAR, rank, disp, sizeof(GPtr), MPI_CHAR, MPI_NO_OP, this->table);
  MPI_Win_flush(rank, this->table);
  return result;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::getBucketData(int bucket, GPtr* arr) {
  int rank = this->getRankByBucket(bucket);
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  int disp = this->getDispBySlot(bucket, 0);
  MPI_Get_accumulate(NULL, 0, MPI_INT, arr, sizeof(GPtr) * 4, MPI_CHAR, rank, disp, sizeof(GPtr) * 4, MPI_CHAR, MPI_NO_OP, this->table);
  MPI_Win_flush(rank, this->table);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline typename BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::DataNode
BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::getData(GPtr node) {
  DataNode result;
  int rank = this->getRankt(node);
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  if (rank >= this->nprocs) {
    std::cout << "[[Rank = " << this->myrank << "]: ??????: " << node << std::endl;
    while (true) {
      sleep(5);
    }
  }
  MPI_Aint disp = this->getDispt(node);
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(DataNode), MPI_CHAR, rank, disp, sizeof(DataNode), MPI_CHAR, MPI_NO_OP, this->data);
  MPI_Win_flush(rank, this->data);
  return result;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::setData(GPtr node, Key key, Value value) {
  DataNode data_node(key, value);
  int rank = this->getRankt(node);
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  MPI_Aint disp = this->getDispt(node);
  MPI_Accumulate(&data_node, sizeof(DataNode), MPI_CHAR, rank, disp, sizeof(DataNode), MPI_CHAR, MPI_REPLACE, this->data);
  MPI_Win_flush(rank, this->data);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline typename BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::GPtr
BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::allocateNode(Key key, Value value) {
  GPtr result = null_gptr;
  if (this->rlist.empty() == false) {
    result = this->rlist.back();
    this->rlist.pop_back();
  } else if (this->plist.empty() == false) {
    result = this->plist.back();
    this->plist.pop_back();
  } else if (this->dlist.empty() == false) {
    this->scan();
    if (this->rlist.empty() == false) {
      result = this->rlist.back();
      this->rlist.pop_back();
    }
  }
  MPI_Aint addr = 0;
  if (result == null_gptr) {
    DataNode* data_node = new DataNode();
    this->data_mem.push_back(data_node);
    MPI_Win_attach(this->data, data_node, sizeof(DataNode));
    MPI_Get_address(data_node, &addr);
    result = this->makeGPtr(this->myrank, addr, 0);
  }
  this->setData(result, key, value);
  return result;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::safeFreeNode(GPtr node) {
  this->dlist.push_back(this->unkickMarked(node));
  if (this->dlist.size() >= this->max_dlist_size)
    this->scan();
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::freeNode(GPtr node) {
  this->plist.push_back(node);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::scan() {
  std::list<GPtr> list_temp;
  std::list<GPtr> dlist_temp;
  GPtr hp;
  for (int i = 0; i < this->nprocs; ++i) {
    hp = this->getHPtr(i, 0);
    if (hp != null_gptr)
      list_temp.push_back(hp);
    hp = this->getHPtr(i, 1);
    if (hp != null_gptr)
      list_temp.push_back(hp);
  }
  list_temp.sort();
  list_temp.unique();
  while (this->dlist.empty() == false) {
    hp = this->dlist.back();
    this->dlist.pop_back();
    if (std::find(list_temp.begin(), list_temp.end(), hp) != list_temp.end())
      dlist_temp.push_back(hp);
    else
      this->rlist.push_back(hp);
  }
  this->dlist.splice(this->dlist.begin(), dlist_temp);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline typename BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::GPtr
BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::makeGPtr(int rank, MPI_Aint disp, int tag) {
  return ((uint64_t)rank << 53) | (disp << 5) | tag;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline int BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::getRankt(GPtr node) {
  return (node >> 53);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline MPI_Aint BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::getDispt(GPtr node) {
  return ((node << 11) >> 16);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline int BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::getRankByBucket(int bucket) {
  if (bucket >= this->table_length_each * this->nprocs) {
    std::cout << "[[Rank = " << this->myrank << "]: ???: " << bucket << std::endl;
    while (true) {
      sleep(5);
    }
  }
  return (bucket / this->table_length_each);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline MPI_Aint BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::getDispBySlot(int bucket, int slot) {
  return (bucket % this->table_length_each) * 4 + slot;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline bool BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::CAS(int bucket, int slot, GPtr old_value, GPtr new_value) {
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]: " << "CAS at <" << bucket << "," << slot << ">" << std::endl;
#endif  // DEBUG
  GPtr result;
  int rank = this->getRankByBucket(bucket);
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  MPI_Aint disp = this->getDispBySlot(bucket, slot);
  MPI_Compare_and_swap(&new_value, &old_value, &result, MPI_UINT64_T, rank, disp, this->table);
  MPI_Win_flush(rank, this->table);
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]: " << "CAS: " << old_value << "->" << new_value << ": " << ((result == old_value) ? "succeed" : "fail") << std::endl;
#else
#ifdef DEBUG2
  if (result != old_value)
    std::cout << "[" << this->myrank << "]: " << "CAS at <" << bucket << "," << slot << ">: " << old_value << "->" << new_value << ": " << "fail" << std::endl;
#endif  // DEBUG2
#endif  // DEBUG
  return (result == old_value);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline typename BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::GPtr
BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::kickMark(GPtr node, int target_slot) {
  return (node | (target_slot << 1) | 1);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline bool BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::isKickMarked(GPtr node) {
  return (node & 1);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline typename BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::GPtr
BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::unkickMarked(GPtr node) {
  return (node & (~7));
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline int BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::getTargetSlot(GPtr node) {
  return ((node >> 1) & 3);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::calBucket(Key key, int& bucket1, int& bucket2) {
  uint64_t hash_value1, hash_value2;
  this->calHash(key, hash_value1, hash_value2);
  this->calBucketFromHash(bucket1, bucket2, hash_value1, hash_value2);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::calHash(Key key, uint64_t& hash_value1, uint64_t& hash_value2) {
#ifdef MURMUR128_SPLIT_C11_HPP
  std::pair<uint64_t, uint64_t> p = MurmurHash3_x64_128_u64(key, 1234);
  hash_value1 = p.first;
  hash_value2 = p.second;
#else
  hash_value1 = this->hash_functor1(key);
  hash_value2 = this->hash_functor2(key);
  if (((hash_value1 - hash_value2) % (this->table_length_each * this->nprocs)) == 0) {
    hash_value2 = this->hash_functor_backup(key);
  }
#endif  // MURMUR128_SPLIT_C11_HPP
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::calHashAndBucket(Key key, uint64_t& hash_value1, uint64_t& hash_value2, int& bucket1, int& bucket2) {
  this->calHash(key, hash_value1, hash_value2);
  this->calBucketFromHash(bucket1, bucket2, hash_value1, hash_value2);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::calBucketFromHash(int& bucket1, int& bucket2, uint64_t& hash_value1, uint64_t& hash_value2) {
  bucket1 = hash_value1 % (this->table_length_each * this->nprocs);
  bucket2 = hash_value2 % (this->table_length_each * this->nprocs);
  if (bucket1 == bucket2) {
    bucket2 = (hash_value2 + 1) % (this->table_length_each * this->nprocs);
  }
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline uint64_t BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::getHpRecord(int rank) {
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  uint64_t result = 2;
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(uint64_t), MPI_CHAR, rank, 0, sizeof(uint64_t), MPI_CHAR, MPI_NO_OP, this->hp_record_win);
  MPI_Win_flush(rank, this->hp_record_win);
  return result;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::setHpRecord(uint64_t hash_value) {
#ifdef COMM_CHECK
  if (isIntra(this->myrank, this->myrank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  MPI_Accumulate(&hash_value, sizeof(uint64_t), MPI_CHAR, this->myrank, 0, sizeof(uint64_t), MPI_CHAR, MPI_REPLACE, this->hp_record_win);
  MPI_Win_flush(this->myrank, this->hp_win);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline uint64_t BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::getHpFlag(int rank) {
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  uint64_t result = 2;
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(uint64_t), MPI_CHAR, rank, 0, sizeof(uint64_t), MPI_CHAR, MPI_NO_OP, this->hp_flag_win);
  MPI_Win_flush(rank, this->hp_flag_win);
  return result;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::setHpFlag(int rank, uint64_t hash_value) {
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  MPI_Accumulate(&hash_value, sizeof(uint64_t), MPI_CHAR, rank, 0, sizeof(uint64_t), MPI_CHAR, MPI_REPLACE, this->hp_flag_win);
  MPI_Win_flush(rank, this->hp_flag_win);
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline typename BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::GPtr
BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::getHPtr(int rank, int number) {
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  GPtr result = 2;
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(GPtr), MPI_CHAR, rank, number, sizeof(GPtr), MPI_CHAR, MPI_NO_OP, this->hp_win);
  MPI_Win_flush(rank, this->hp_win);
  return result;
}
template <typename Key, typename Value, typename HashFunctor1, typename HashFunctor2, typename HashFunctor3>
inline void BucketizedCuckooHashTable<Key, Value, HashFunctor1, HashFunctor2, HashFunctor3>::setHPtr(GPtr node, int number) {
#ifdef COMM_CHECK
  if (isIntra(this->myrank, this->myrank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  GPtr no_mark_node = this->unkickMarked(node);
  MPI_Accumulate(&no_mark_node, sizeof(GPtr), MPI_CHAR, this->myrank, number, sizeof(GPtr), MPI_CHAR, MPI_REPLACE, this->hp_win);
  MPI_Win_flush(this->myrank, this->hp_win);
}
}  // namespace ngu

#endif  // DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_BUCKETIZED_CUCKOO_HASHTABLE_H