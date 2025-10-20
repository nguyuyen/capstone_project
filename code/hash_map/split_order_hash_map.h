#ifndef DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_SPLIT_ORDER_HASH_MAP_H
#define DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_SPLIT_ORDER_HASH_MAP_H

// #define COMM_CHECK

#ifdef DEBUG
#include <unistd.h>
#endif  // DEBUG

#include <mpi.h>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <cstring>
#include <functional>
#include <iostream>
#include <list>
#include <random>

namespace ngu {
template <typename T>
using Bitsets = std::bitset<sizeof(T) * 8>;

std::pair<uint32_t, uint32_t> locate(uint32_t n, uint32_t base) {
  // base là kích thước của a0 (1, 2, 4, 8, ...)
  if (n < base) return {0u, n};
  if (n < 2 * base) return {1u, n - base};

  // n >= 2*base
  uint32_t x = n / base;                    // quy về thang 2^k
  uint32_t log2x = 31u - __builtin_clz(x);  // floor(log2(x))
  uint32_t i = log2x + 1;                   // i = floor(log2(n/base)) + 1
  uint32_t start = base << (i - 1);         // S(i) = base * 2^(i-1)
  uint32_t j = n - start;
  return {i, j};
}

/// @brief Make bitset from data.
/// @tparam T Type of data.
/// @param data Data.
/// @return Result bitset.
template <typename T>
Bitsets<T> toBitsets(const T& data) {
  Bitsets<T> bits;
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&data);
  for (size_t i = 0; i < sizeof(T); i++) {
    for (size_t j = 0; j < 8; j++) {
      bits[i * 8 + j] = (ptr[i] >> j) & 1;
    }
  }
  return bits;
}

/// @brief Reverse bitset.
/// @tparam N Size of input bitset.
/// @param bits Input bitset.
template <size_t N>
void reverseBitset(std::bitset<N>& bits) {
  for (size_t i = 0; i < N / 2; ++i) {
    bool tmp = bits[i];
    bits[i] = bits[N - 1 - i];
    bits[N - 1 - i] = tmp;
  }
}

template <typename Key>
class Hash5 {
 public:
  Bitsets<Key> operator()(Key key) const {
    Bitsets<Key> bits = toBitsets(key);
    reverseBitset(bits);
    return bits;
  };
};

template <size_t N>
uint32_t getLast32Bits(const std::bitset<N>& bits) {
  uint32_t result = 0;
  for (size_t i = 0; i < ((N > 32) ? 32 : N); ++i) {
    if (bits[i]) result |= (1u << i);
  }
  return result;
}

template <typename T>
uint32_t get_last32(const T& obj) {
  uint32_t last32 = 0;
  size_t size = sizeof(T);

  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&obj);
  std::memcpy(&last32, ptr + size - sizeof(uint32_t), sizeof(uint32_t));

  return last32;
}

template <typename Key>
class Compare2 {
 public:
  bool operator()(Bitsets<Key> bs1, Bitsets<Key> bs2) const {  // bs1 < bs2
    for (int i = bs1.size() - 1; i >= 0; --i) {
      if (bs1[i] ^ bs2[i]) {
#ifdef DEBUG
        std::cout << "Compare: " << bs1 << "<" << bs2 << ": " << ((bs2[i] == true) ? "true" : "false") << std::endl;
#endif  // DEBUG
        return bs2[i];
      }
    }
#ifdef DEBUG
    std::cout << "Compare: " << bs1 << "," << bs2 << ": " << "false" << std::endl;
#endif  // DEBUG
    return false;
  };
};

template <typename Key, typename Value, typename HashFunctor = Hash5<Key>, typename KeyCompare = Compare2<Key>>
class SplitOrderHashMap {
  using HashValue = Bitsets<Key>;

  /// @brief rank(15 bit) - disp(48 bit) - tag(1 bit)
  using GPtr = uint64_t;
  class DataNode;

 private:
  static const GPtr null_gptr = 0;

 public:
  SplitOrderHashMap(MPI_Comm comm = MPI_COMM_WORLD, int target_load_factor_in_percent = 200, uint32_t segment_base_size = 4);
  SplitOrderHashMap(const SplitOrderHashMap&) = delete;
  SplitOrderHashMap& operator=(const SplitOrderHashMap&) = delete;
  ~SplitOrderHashMap();
  /// @brief Insert a key-value pair into hash map.
  /// @param key An input key.
  /// @param value An input value.
  /// @return True if the key is not in the hash map, False if the key is already there.
  bool insert(Key key, Value value);
  /// @brief Traverse the hash map.
  /// @param key An input key.
  /// @param value Output value.
  /// @return True if found, otherwise false.
  bool get(Key key, Value& value);
  /// @brief Remove a key-value pair that is present in the hash map.
  /// @param key The key has value to remove.
  /// @return True if remove successfully, False if the key is not present in the hash map.
  bool remove(Key key);

  // #ifdef MEM_CHECK
  int getMem() {
    return (sizeof(GPtr) * this->total_segment_size +  // segment
            sizeof(DataNode) * data_mem.size());       // data
  }
  int getLoadFactorInPercent() {
    return this->getCount() * 100 / this->getSize();
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
  bool insertToList(GPtr head, HashValue hash_value, Value value);
  bool getFromList(GPtr head, HashValue hash_value, Value& value);
  bool removeFromList(GPtr head, HashValue hash_value);
  /// @brief Find node from hash value in list.
  /// @param head Head of list.
  /// @param hash_value Input hash value.
  /// @param value Output value if key found.
  /// @param prev Ouput prev node of node has key if found.
  /// @param curr Ouput node has key if found.
  /// @param next Ouput next node of node has key if found.
  /// @return True if found, otherwise False.
  bool findInList(GPtr head, HashValue hash_value, Value& value, GPtr& prev, GPtr& curr, GPtr& next);
  /// @brief Get data node.
  /// @param node Address.
  /// @return Data node.
  DataNode getData(GPtr node);
  /// @brief Set data node.
  /// @param node Address.
  /// @param next Next node.
  /// @param key New key.
  /// @param value New value.
  void setData(GPtr node, GPtr next, HashValue hash_value, Value value);
  /// @brief Ensure momory is not freed while another process is using it.
  /// @param node_to_free Node to free
  void safeFreeNode(GPtr node_to_free);
  /// @brief Free a node without protect.
  /// @param node_to_free Node to free.
  void freeNode(GPtr node_to_free);
  /// @brief Allocate data node, reuse memory before alloc new.
  /// @param value Value.
  /// @param hash_value Hash value.
  /// @return Pointer to this node.
  GPtr allocateNode(HashValue hash_value, Value value, GPtr next = null_gptr);
  /// @brief Scan deleted node for reuse.
  void scan();

 private:
  HashValue calSplitOrderRegular(Key key);
  HashValue calSplitOrderDummy(uint32_t bucket);
  void initBucket(int position);
  GPtr getSegment(int position);
  GPtr getBucketInSegment(GPtr segment, int position);
  void setHeadNode(int position, GPtr head);
  uint32_t getSize();
  void doubleSize(uint32_t old_size);
  uint32_t getCount();
  uint32_t increaseCount();
  uint32_t decreaseCount();
  uint32_t getParentBucket(uint32_t bucket);
  /// @brief Get hazard pointer.
  /// @param rank Rank.
  /// @param number N-th hazard pointer.
  /// @return Pointer.
  GPtr getHPtr(int rank, int number);
  /// @brief Mark a node as hazard
  /// @param node Address
  /// @param number N-th hazard pointer.
  void setHPtr(GPtr node, int number);
  GPtr makeGPtr(int rank, MPI_Aint disp, int tag);
  /// @brief Get head node from position in head array.
  /// @param position Position
  /// @return Address
  GPtr getHeadNode(int position);
  int getRank(GPtr node);
  MPI_Aint getDisp(GPtr node);
  /// @brief Compare And Swap: swap next of a node to new value.
  /// @param curr Address of current node.
  /// @param old_next Address of next of current node.
  /// @param new_next Address of new next of current node.
  /// @return
  bool CASNext(GPtr curr, GPtr old_next, GPtr new_next);
  /// @brief Get Address of node with mark.
  /// @param node Address of node.
  /// @return Address of node after mark.
  GPtr mark(GPtr node);
  /// @brief Get Address of node without mark.
  /// @param node Address of node.
  /// @return Address of node after unmark.
  GPtr unmark(GPtr node);
  /// @brief Get next node of a node.
  /// @param node Address of node.
  /// @return Address of next node.
  GPtr getNextNode(GPtr node);
  bool isMarked(GPtr node);

 private:
  int nprocs;
  int myrank;
  int target_load_factor_in_percent;
  HashFunctor hash_functor;
  KeyCompare key_order;
  std::size_t max_dlist_size = 10;
  uint32_t* count_size;  // total item count and current size
  uint32_t segment_base_size = 4;

  uint32_t dic_arr_size_each = 32;

  uint16_t total_segment_size = 0;

#ifdef COMM_CHECK
  bool isIntra(int source_rank, int target_rank) {
    return (source_rank / 8) == (target_rank / 8);
  }
  int intra_comm_count = 0;
  int inter_comm_count = 0;
#endif  // COMM_CHECK

 private:
  MPI_Comm comm;
  MPI_Win data;
  MPI_Win dic_win;
  MPI_Win size_win;
  MPI_Win segment_win;

 private:
  std::list<DataNode*> data_mem;
  MPI_Win hp_win;
  GPtr* hp_arr;
  std::list<GPtr> plist;
  std::list<GPtr> rlist;
  std::list<GPtr> dlist;
  GPtr* dic_arr;
  std::list<GPtr*> segment_list;

  class DataNode {
    using GPtr = SplitOrderHashMap::GPtr;

   public:
    DataNode() : next(), hash_value(), value() {};
    DataNode(HashValue hash_value, Value value, GPtr next = SplitOrderHashMap::null_gptr)
        : next(next), hash_value(hash_value), value(value) {};

   public:
    GPtr next;
    HashValue hash_value;
    Value value;
  };
};

template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::SplitOrderHashMap(MPI_Comm comm, int target_load_factor_in_percent, uint32_t segment_base_size)
    : target_load_factor_in_percent(target_load_factor_in_percent), comm(comm), data_mem(), plist(), rlist(), dlist() {
  if (segment_base_size < 4) {
    this->segment_base_size = 4;
  } else {
    this->segment_base_size = 1u << (31 - __builtin_clz(segment_base_size));
  }

  MPI_Comm_size(comm, &this->nprocs);
  MPI_Comm_rank(comm, &this->myrank);
#ifdef DEBUG
  std::cout << "[[" << this->myrank << "]]" << ": constructing..." << std::endl;
#endif  // DEBUG

  MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &this->data);

  this->hp_arr = new GPtr[3]();
  MPI_Win_create(this->hp_arr, sizeof(GPtr) * 3, sizeof(GPtr), MPI_INFO_NULL, comm, &this->hp_win);

  this->dic_arr = new GPtr[this->dic_arr_size_each]();
  MPI_Win_create(this->dic_arr, sizeof(GPtr) * dic_arr_size_each, sizeof(GPtr), MPI_INFO_NULL, comm, &this->dic_win);

  MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &this->segment_win);

  this->count_size = new uint32_t[2]{4, 0};
  MPI_Win_create(this->count_size, sizeof(uint32_t) * 2, sizeof(uint32_t), MPI_INFO_NULL, comm, &this->size_win);
#ifdef DEBUG_HASH
  if (this->myrank == 0) {
    int temp_bucket1;
    int number_key_try = (this->table_length_each * this->nprocs) * 100;
    int number_of_bucket = this->table_length_each * this->nprocs;
    int* count_bucket_hit = new int[number_of_bucket]();
    for (int i = 0; i < number_key_try; ++i) {
      temp_bucket1 = hash_functor(i) % (this->table_length_each * this->nprocs);
      ++(count_bucket_hit[temp_bucket1]);
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

  MPI_Win_fence(0, this->data);
  MPI_Win_fence(0, this->hp_win);
  MPI_Win_fence(0, this->dic_win);
  MPI_Win_fence(0, this->segment_win);
  MPI_Win_fence(0, this->size_win);
  MPI_Win_lock_all(0, this->data);
  MPI_Win_lock_all(0, this->hp_win);
  MPI_Win_lock_all(0, this->dic_win);
  MPI_Win_lock_all(0, this->segment_win);
  MPI_Win_lock_all(0, this->size_win);

  if (this->myrank == 0) {
    GPtr dummy0 = this->allocateNode(this->calSplitOrderDummy(0), {}, null_gptr);
    this->setHeadNode(0, dummy0);
  }

  MPI_Barrier(comm);
#ifdef DEBUG
  std::cout << "[[" << this->myrank << "]]" << ": construct successfully" << std::endl;
#endif  // DEBUG
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::~SplitOrderHashMap() {
  int flag;
  for (typename std::list<DataNode*>::iterator it = data_mem.begin(); it != data_mem.end(); ++it) {
    delete *it;
  }
  delete[] this->hp_arr;
  delete[] this->dic_arr;
  for (typename std::list<GPtr*>::iterator it = segment_list.begin(); it != segment_list.end(); ++it) {
    delete[] *it;
  }
  delete[] this->count_size;
  MPI_Finalized(&flag);
  if (!flag) {
    MPI_Win_unlock_all(this->data);
    MPI_Win_unlock_all(this->hp_win);
    MPI_Win_unlock_all(this->dic_win);
    MPI_Win_unlock_all(this->segment_win);
    MPI_Win_unlock_all(this->size_win);
    MPI_Win_free(&this->data);
    MPI_Win_free(&this->hp_win);
    MPI_Win_free(&this->dic_win);
    MPI_Win_free(&this->segment_win);
    MPI_Win_free(&this->size_win);
  }
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::insert(Key key, Value value) {
  int position = get_last32(key) % this->getSize();
#ifdef DEBUG
  std::cout << "[[Rank = " << this->myrank << "]]: " << "insert <" << key << "," << value << ">. at: " << position << std::endl;
#endif  // DEBUG
  GPtr head_node = this->getHeadNode(position);
  if (head_node == null_gptr) {
    this->initBucket(position);
  }
  head_node = this->getHeadNode(position);
  if (this->insertToList(head_node, this->calSplitOrderRegular(key), value) == false) {
    return false;
  } else {
    uint32_t new_count = this->increaseCount();
    uint32_t size_now = this->getSize();
    if ((new_count * 100.0 / size_now) > (this->target_load_factor_in_percent * 1.25)) {  // current load factor > 1.2 * target
#ifdef DEBUG_EXPAND
      std::cout << "--[" << this->myrank << "]: " << "need expand : count = " << new_count << "/" << size_now << ", target: " << this->target_load_factor_in_percent << "/100" << std::endl;
#endif  // DEBUG_EXPAND
      this->doubleSize(size_now);
    }
    return true;
  }
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::get(Key key, Value& value) {
  int position = get_last32(key) % this->getSize();
  GPtr head_node = this->getHeadNode(position);
  if (head_node == null_gptr) {
    this->initBucket(position);
  }
  head_node = this->getHeadNode(position);
  return this->getFromList(head_node, this->calSplitOrderRegular(key), value);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::remove(Key key) {
  int position = get_last32(key) % this->getSize();
  GPtr head_node = this->getHeadNode(position);
  if (head_node == null_gptr) {
    this->initBucket(position);
  }
  head_node = this->getHeadNode(position);
  if (this->removeFromList(head_node, this->calSplitOrderRegular(key)) == false) {
    return false;
  } else {
    this->decreaseCount();
    return true;
  }
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::insertToList(GPtr head, HashValue hash_value, Value value) {
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "insert node at head node: " << head << ", " << hash_value << std::endl;
#endif  // DEBUG
  bool result = false;
  Value value_temp;
  GPtr prev = null_gptr, curr = null_gptr, next = null_gptr;
#ifdef DEBUG
  int debug_count = 0;
  int debug_max_count = 10;
#endif  // DEBUG
  while (true) {
#ifdef DEBUG
    ++debug_count;
    if (debug_count >= debug_max_count) {
      if (debug_count == debug_max_count) {
        std::cout << "[Rank = " << this->myrank << "]: " << "insert node to list fail loop 1: " << prev << " -> " << curr << std::endl;
      }
      MPI_Abort(this->comm, 1);
    }
#endif  // DEBUG
    if (findInList(head, hash_value, value_temp, prev, curr, next) == true) {
      result = false;
      break;
    }
    GPtr new_node = this->allocateNode(hash_value, value, unmark(curr));
    if (CASNext(prev, unmark(curr), unmark(new_node)) == true) {
      result = true;
#ifdef DEBUG2
      std::cout << this->myrank << ": " << key << std::endl;
#endif  // DEBUG2
      break;
    } else
      this->freeNode(unmark(new_node));
  }
  this->setHPtr(null_gptr, 0);
  this->setHPtr(null_gptr, 1);
  this->setHPtr(null_gptr, 2);
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "insert at head success." << std::endl;
#endif  // DEBUG
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::getFromList(GPtr head, HashValue hash_value, Value& value) {
  GPtr prev = null_gptr, curr = null_gptr, next = null_gptr;
  bool result = this->findInList(head, hash_value, value, prev, curr, next);
  this->setHPtr(null_gptr, 0);
  this->setHPtr(null_gptr, 1);
  this->setHPtr(null_gptr, 2);
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::removeFromList(GPtr head, HashValue hash_value) {
  bool result = false;
  Value value_temp;
  GPtr prev = null_gptr, curr = null_gptr, next = null_gptr;
#ifdef DEBUG
  int debug_count = 0;
  int debug_max_count = 10;
#endif  // DEBUG
  while (true) {
#ifdef DEBUG
    ++debug_count;
    if (debug_count >= debug_max_count) {
      if (debug_count == debug_max_count) {
        std::cout << "[Rank = " << this->myrank << "]: " << "remove node from list fail loop 1: " << prev << " -> " << curr << std::endl;
      }
      MPI_Abort(this->comm, 1);
    }
#endif  // DEBUG
    if (this->findInList(head, hash_value, value_temp, prev, curr, next) == false) {
      result = false;
      break;
    }
    if (CASNext(curr, unmark(next), mark(next)) == false)
      continue;
    if (CASNext(prev, unmark(curr), unmark(next)) == true)
      this->safeFreeNode(unmark(curr));
    else
      this->findInList(head, hash_value, value_temp, prev, curr, next);
    result = true;
    break;
  }
  this->setHPtr(null_gptr, 0);
  this->setHPtr(null_gptr, 1);
  this->setHPtr(null_gptr, 2);
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::findInList(GPtr head, HashValue hash_value, Value& value, GPtr& prev, GPtr& curr, GPtr& next) {
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "get node from list node... " << std::endl;
#endif  // DEBUG
#ifdef DEBUG
  int debug_count = 0;
  int debug_max_count = 10;
  int debug_count2 = 0;
#endif  // DEBUG
  bool flag = false;
  while (true) {
#ifdef DEBUG
    debug_count2 = 0;
    ++debug_count;
    if (debug_count >= debug_max_count) {
      if (debug_count == debug_max_count) {
        std::cout << "[Rank = " << this->myrank << "]: " << "get node from list node fail loop 1: " << prev << " -> " << curr << std::endl;
      }
      MPI_Abort(this->comm, 1);
    }
#endif  // DEBUG
    flag = false;
    prev = head;
    curr = this->getNextNode(prev);
    this->setHPtr(unmark(curr), 1);
    if (unmark(curr) != this->getNextNode(prev)) {
#ifdef DEBUG
      std::cout << "[Rank = " << this->myrank << "]: " << " node changed: " << curr << "," << unmark(curr) << std::endl;
#endif  // DEBUG
      continue;
    }
    while (true) {
#ifdef DEBUG
      ++debug_count2;
      if (debug_count2 >= debug_max_count) {
        if (debug_count2 == debug_max_count) {
          std::cout << "[Rank = " << this->myrank << "]: " << "get node from list node fail loop 2: " << prev << " -> " << curr << std::endl;
        }
        MPI_Abort(this->comm, 1);
      }
#endif  // DEBUG
      if (unmark(curr) == null_gptr) {
#ifdef DEBUG
        std::cout << "[Rank = " << this->myrank << "]: " << "get node from list node null." << std::endl;
#endif  // DEBUG
        return false;
      }
      next = this->getNextNode(curr);
      this->setHPtr(unmark(next), 0);
      if (next != this->getNextNode(curr)) {
        flag = true;
        break;
      }
      DataNode data_node = this->getData(curr);
      HashValue curr_hash_value = data_node.hash_value;
      value = data_node.value;
      if (unmark(curr) != this->getNextNode(prev)) {
        flag = true;
        break;
      }
      if (isMarked(next) == false) {
        if (this->key_order(curr_hash_value, hash_value) == false) {  // curr_hash_value >= hash_value
#ifdef DEBUG
          std::cout << "[Rank = " << this->myrank << "]: " << "get node from list node return: " << (curr_hash_value == hash_value) << "||" << prev << " -> " << curr << std::endl;
#endif  // DEBUG
          return (curr_hash_value == hash_value);
        }
#ifdef DEBUG
        std::cout << "[Rank = " << this->myrank << "]: " << "get node from list goto next node:" << prev << " -> " << curr << std::endl;
#endif  // DEBUG
        prev = unmark(curr);
        this->setHPtr(unmark(curr), 2);
      } else {
        if (CASNext(prev, unmark(curr), unmark(next)) == true)
          this->safeFreeNode(unmark(curr));
        else {
          flag = true;
          break;
        }
      }
      curr = unmark(next);
      this->setHPtr(unmark(next), 1);
    }
    if (flag == true)
      continue;
  }
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::DataNode
SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::getData(GPtr node) {
  DataNode result;
  int rank = this->getRank(node);
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  MPI_Aint disp = this->getDisp(node);
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(DataNode), MPI_CHAR, rank, disp, sizeof(DataNode), MPI_CHAR, MPI_NO_OP, this->data);
  MPI_Win_flush(rank, this->data);
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::setData(GPtr node, GPtr next, HashValue hash_value, Value value) {
  DataNode data_node(hash_value, value, next);
  int rank = this->getRank(node);
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  MPI_Aint disp = this->getDisp(node);
  MPI_Accumulate(&data_node, sizeof(DataNode), MPI_CHAR, rank, disp, sizeof(DataNode), MPI_CHAR, MPI_REPLACE, this->data);
  MPI_Win_flush(rank, this->data);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::safeFreeNode(GPtr node_to_free) {
  this->dlist.push_back(node_to_free);
  if (this->dlist.size() >= this->max_dlist_size)
    this->scan();
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::freeNode(GPtr node_to_free) {
  this->plist.push_back(node_to_free);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::GPtr
SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::allocateNode(HashValue hash_value, Value value, GPtr next) {
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "alloc... hash_value: " << hash_value << std::endl;
#endif  // DEBUG
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
  this->setData(result, next, hash_value, value);
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "alloc success: " << result << std::endl;
#endif  // DEBUG
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::scan() {
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
    hp = this->getHPtr(i, 2);
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
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::HashValue
SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::calSplitOrderRegular(Key key) {
  HashValue bits = toBitsets(key);
  reverseBitset(bits);
  bits.set(0);
#ifdef DEBUG5
  std::cout << "[" << this->myrank << "]: " << "regular: " << bits << std::endl;
#endif  // DEBUG5
  return bits;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::HashValue
SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::calSplitOrderDummy(uint32_t bucket) {
  HashValue bits(bucket);
  reverseBitset(bits);
  return bits;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::initBucket(int position) {
#ifdef DEBUG5
  std::cout << "---[Rank = " << this->myrank << "]: " << "init bucket " << position << std::endl;
  if (position == 0) {
    MPI_Abort(comm, 1);
  }
#endif  // DEBUG5
  uint16_t parent = this->getParentBucket(position);
  GPtr parent_head = this->getHeadNode(parent);
  if (parent_head == null_gptr) {
    this->initBucket(parent);
  }
  parent_head = this->getHeadNode(parent);
  HashValue dummy = this->calSplitOrderDummy(position);
  Value useless_value;
  this->insertToList(parent_head, dummy, {});
  GPtr prev = null_gptr, curr = null_gptr, next = null_gptr;
  if (this->findInList(parent_head, dummy, useless_value, prev, curr, next) == true) {
    this->setHeadNode(position, curr);
  } else {
    std::cout << "---[Rank = " << this->myrank << "]: " << "init bucket fail." << std::endl;
    MPI_Abort(this->comm, 1);
  }
#ifdef DEBUG5
  std::cout << "---[Rank = " << this->myrank << "]: " << "init bucket " << position << " done" << std::endl;
#endif  // DEBUG5
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::GPtr
SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::getSegment(int position) {
  // rank[k]: segment(k), segment(k+n), segment(k+2n),...
  int rank = position % this->nprocs;
  int disp = position / this->nprocs;
  GPtr result = 1;
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(GPtr), MPI_CHAR, rank, disp, sizeof(GPtr), MPI_CHAR, MPI_NO_OP, this->dic_win);
  MPI_Win_flush(rank, this->dic_win);
#ifdef DEBUG5
  std::cout << "[Rank = " << this->myrank << "]: " << "get segment at " << position << ": " << result << std::endl;
#endif  // DEBUG5
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::GPtr
SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::getBucketInSegment(GPtr segment, int position) {
#ifdef DEBUG5
  std::cout << "[Rank = " << this->myrank << "]: " << "get bucket in segment at " << segment << ", index:" << position << std::endl;
#endif  // DEBUG5
  int rank = this->getRank(segment);
  MPI_Aint disp = this->getDisp(segment) + position * sizeof(GPtr);
  GPtr result = 1;
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(GPtr), MPI_CHAR, rank, disp, sizeof(GPtr), MPI_CHAR, MPI_NO_OP, this->segment_win);
  MPI_Win_flush(rank, this->segment_win);
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::setHeadNode(int position, GPtr head) {
#ifdef DEBUG5
  std::cout << "[Rank = " << this->myrank << "]: " << "set head node at " << position << "->" << head << std::endl;
#endif  // DEBUG5
  std::pair<uint32_t, uint32_t> p = locate(position, this->segment_base_size);
  GPtr segment = this->getSegment(p.first);
  GPtr null_head = null_gptr;
  if (segment == null_gptr) {
    int rank = p.first % this->nprocs;
    int disp = p.first / this->nprocs;
    uint32_t new_segment_size = ((p.first < 2) ? segment_base_size : (segment_base_size << (p.first - 1)));
    GPtr* new_seg_arr = new GPtr[new_segment_size]();
    MPI_Win_attach(this->segment_win, new_seg_arr, new_segment_size * sizeof(GPtr));
    MPI_Aint addr;
    MPI_Get_address(new_seg_arr, &addr);
    GPtr a_node = this->makeGPtr(this->myrank, addr, 0);
    GPtr result = null_gptr;
    MPI_Compare_and_swap(&a_node, &null_head, &result, MPI_UINT64_T, rank, disp, this->dic_win);
    MPI_Win_flush(rank, this->dic_win);
    if (result != null_gptr) {
#ifdef DEBUG5
      std::cout << "[Rank = " << this->myrank << "]: " << "set segment node at " << position << ": CAS fail" << std::endl;
#endif  // DEBUG5
      MPI_Win_detach(this->segment_win, new_seg_arr);
      delete[] new_seg_arr;
    } else {
      this->segment_list.push_back(new_seg_arr);
      this->total_segment_size += new_segment_size;
#ifdef DEBUG5
      std::cout << "[Rank = " << this->myrank << "]: " << "set segment length: " << new_segment_size << ", at " << position << ": CAS succeed" << std::endl;
#endif  // DEBUG5
    }
  }
  GPtr result = null_gptr;
  segment = this->getSegment(p.first);
  int rank = this->getRank(segment);
  MPI_Aint disp = this->getDisp(segment) + p.second * sizeof(GPtr);
  MPI_Compare_and_swap(&head, &null_head, &result, MPI_UINT64_T, rank, disp, this->segment_win);
#ifdef DEBUG5
  if (result == null_head) {
    std::cout << "[Rank = " << this->myrank << "]: " << "CAS head node at " << position << ": CAS succeed" << std::endl;
  } else {
    std::cout << "[Rank = " << this->myrank << "]: " << "CAS head node at " << position << ": CAS fail" << std::endl;
  }
#endif  // DEBUG5
  MPI_Win_flush(rank, this->segment_win);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline uint32_t SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::getSize() {
  uint32_t result = 2;
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(uint32_t), MPI_CHAR, 0, 0, sizeof(uint32_t), MPI_CHAR, MPI_NO_OP, this->size_win);
  MPI_Win_flush(0, this->size_win);
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::doubleSize(uint32_t old_size) {
  uint32_t new_size = old_size * 2;
  uint32_t result;
  MPI_Compare_and_swap(&new_size, &old_size, &result, MPI_UINT32_T, 0, 0, this->size_win);
  MPI_Win_flush(0, this->size_win);

#ifdef DEBUG_EXPAND
  if (result == old_size) {
    std::cout << "[" << this->myrank << "]: " << "expand : " << old_size << "->" << new_size << std::endl;
  }
#endif  // DEBUG_EXPAND
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline uint32_t SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::getCount() {
  uint32_t result = 2;
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(uint32_t), MPI_CHAR, 0, 1, sizeof(uint32_t), MPI_CHAR, MPI_NO_OP, this->size_win);
  MPI_Win_flush(0, this->size_win);
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline uint32_t SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::increaseCount() {
  uint32_t m = 1;
  uint32_t result;
  MPI_Fetch_and_op(&m, &result, MPI_UINT32_T, 0, 1, MPI_SUM, this->size_win);
  MPI_Win_flush(0, this->size_win);
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]: " << "increase count to: " << result + 1 << std::endl;
#endif  // DEBUG
  return result + 1;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline uint32_t SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::decreaseCount() {
  int32_t m = -1;
  uint32_t result;
  MPI_Fetch_and_op(&m, &result, MPI_UINT32_T, 0, 1, MPI_SUM, this->size_win);
  MPI_Win_flush(0, this->size_win);
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]: " << "decrease count to: " << result - 1 << std::endl;
#endif  // DEBUG
  return result - 1;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline uint32_t SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::getParentBucket(uint32_t bucket) {
  if (bucket == 0) return 0;
  // tìm bit 1 cao nhất
  uint32_t highest = 1u << (31 - __builtin_clz(bucket));
  return bucket ^ highest;  // hoặc: x & ~highest;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::GPtr
SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::getHPtr(int rank, int number) {
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
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::setHPtr(GPtr node, int number) {
#ifdef COMM_CHECK
  if (isIntra(this->myrank, this->myrank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  MPI_Accumulate(&node, sizeof(GPtr), MPI_CHAR, this->myrank, number, sizeof(GPtr), MPI_CHAR, MPI_REPLACE, this->hp_win);
  MPI_Win_flush(this->myrank, this->hp_win);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::GPtr
SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::makeGPtr(int rank, MPI_Aint disp, int tag) {
  return ((uint64_t)rank << 49) | (disp << 1) | tag;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::GPtr
SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::getHeadNode(int position) {
  std::pair<uint32_t, uint32_t> p = locate(position, this->segment_base_size);
#ifdef DEBUG5
  std::cout << "[Rank = " << this->myrank << "]: " << "get head node at " << position << ": <" << p.first << "," << p.second << ">" << std::endl;
#endif  // DEBUG5
  GPtr segment = this->getSegment(p.first);
  if (segment == null_gptr) {
    return null_gptr;
  } else {
    return this->getBucketInSegment(segment, p.second);
  }
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline int SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::getRank(GPtr node) {
  return (node >> 49);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline MPI_Aint SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::getDisp(GPtr node) {
  // return ((node << 15) >> 16);
  return (node >> 1) & ((1ULL << 48) - 1);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::CASNext(GPtr curr, GPtr old_next, GPtr new_next) {
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]: " << ": CAS at: " << curr << ", " << old_next << "->" << new_next << std::endl;
#endif  // DEBUG
  GPtr result;
  int rank = this->getRank(curr);
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  MPI_Aint disp = this->getDisp(curr);
  MPI_Compare_and_swap(&new_next, &old_next, &result, MPI_UINT64_T, rank, disp, this->data);
  MPI_Win_flush(rank, this->data);
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]: " << ": CAS result: " << ((result == old_next) ? "succeed" : "fail") << std::endl;
#else
#ifdef DEBUG3
  std::cout << "[" << this->myrank << "]: " << ": CAS at: " << curr << ", " << old_next << "->" << new_next << "." << ((result == old_next) ? "succeed" : "fail") << std::endl;
#endif  // DEBUG3
#endif  // DEBUG
  return (result == old_next);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::GPtr
SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::mark(GPtr node) {
  return (node | 1);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::GPtr
SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::unmark(GPtr node) {
  return (node & (~1));
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::GPtr
SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::getNextNode(GPtr node) {
  GPtr result;
  int rank = this->getRank(node);
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  MPI_Aint disp = this->getDisp(node);
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(GPtr), MPI_CHAR, rank, disp, sizeof(GPtr), MPI_CHAR, MPI_NO_OP, this->data);
  MPI_Win_flush(rank, this->data);
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "get next node: " << node << " -> " << result << std::endl;
#endif  // DEBUG
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool SplitOrderHashMap<Key, Value, HashFunctor, KeyCompare>::isMarked(GPtr node) {
  return (node & 1);
}
}  // namespace ngu

#endif  // DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_SPLIT_ORDER_HASH_MAP_H