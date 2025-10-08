#ifndef DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_WAIT_FREE_HASH_MAP_H
#define DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_WAIT_FREE_HASH_MAP_H

#define BASIC_HASH_MAP

// #define COMM_CHECK

#include <mpi.h>
#include <unistd.h>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <iostream>
#include <list>
#include <new>
#include <random>

namespace ngu {

template <typename T>
using Bitset = std::bitset<sizeof(T) * 8>;

/// @brief Make bitset from data.
/// @tparam T Type of data.
/// @param data Data.
/// @return Result bitset.
template <typename T>
Bitset<T> toBitset(const T& data) {
  Bitset<T> bits;
  const uint8_t* ptr = reinterpret_cast<const uint8_t*>(&data);
  for (size_t i = 0; i < sizeof(T); i++) {
    for (size_t j = 0; j < 8; j++) {
      bits[i * 8 + j] = (ptr[i] >> j) & 1;
    }
  }
  return bits;
}

/// @brief Shuffle bitset.
/// @tparam N Size of input bitset.
/// @param bits Input bitset.
/// @param seed Seed.
template <size_t N>
void shuffle_bitset(std::bitset<N>& bits, int seed = 2510) {
  std::mt19937 g(seed);
  for (size_t i = N - 1; i > 0; --i) {
    std::uniform_int_distribution<size_t> dist(0, i);
    size_t j = dist(g);
    if (i != j) {
      bool tmp = bits[i];
      bits[i] = bits[j];
      bits[j] = tmp;
    }
  }
}

/// @brief Hash functor: Hash key to bitset.
/// @tparam Key Key
template <typename Key>
class Hash {
 public:
  Bitset<Key> operator()(Key key) const {
    Bitset<Key> bits = toBitset(key);
#ifndef DEBUG
    shuffle_bitset(bits);
#endif  // DEBUG
    return bits;
  };
};

/// @brief Distributed version of Laborde, P., Feldman, S., & Dechev, D. (2017). A wait-free hash map. International Journal of Parallel Programming, 45, 421-448.
/// @tparam Key Key Type
/// @tparam Value Value Type
/// @tparam HashFunctor Hash Functor Type
template <typename Key, typename Value, typename HashFunctor = Hash<Key>>
class WaitFreeHashMap {
  using HashValue = Bitset<Key>;

  /// @brief rank(14 bit) - disp(48 bit) - tag(2 bit)
  using GPtr = uint64_t;
  class DataNode;

 private:
  static const GPtr null_gptr = 0;

 public:
  WaitFreeHashMap(MPI_Comm comm = MPI_COMM_WORLD);
  WaitFreeHashMap(const WaitFreeHashMap&) = delete;
  WaitFreeHashMap& operator=(const WaitFreeHashMap&) = delete;
  ~WaitFreeHashMap();
  /// @brief Insert a key-value pair into hash map.
  /// @param key An input key.
  /// @param value An input value.
  /// @return True if the key is not in the hash map, False if the key is already there.
  bool insert(Key key, Value value);

#ifdef BASIC_HASH_MAP
#else
  /// @brief Update the value associated with a key that is present in the hash map.
  /// @param key The key has value to update.
  /// @param expected_value The value expected to be associated with this key.
  /// @param new_value The value to associate with this key.
  /// @return True if replace successfully, False if the key is not present in the hash map or if the key associated value does not match expected.
  bool update(Key key, Value expected_value, Value new_value);
#endif  // BASIC_HASH_MAP

#ifdef BASIC_HASH_MAP
  /// @brief Traverse the hash map.
  /// @param key An input key.
  /// @param value Output value.
  /// @return True if found, otherwise false.
  bool get(Key key, Value& value);
#else
  /// @brief Traverse the hash map.
  /// @param key An input key.
  /// @return The value associated with the key if key match, otherwise null.
  Value get(Key key);
#endif  // BASIC_HASH_MAP

#ifdef BASIC_HASH_MAP
  /// @brief Remove a key-value pair that is present in the hash map.
  /// @param key The key has value to remove.
  /// @return True if remove successfully, False if the key is not present in the hash map.
  bool remove(Key key);
#else
  /// @brief Remove a key-value pair that is present in the hash map.
  /// @param key The key has value to remove.
  /// @param expected_value The value expected to be associated with this key.
  /// @return True if remove successfully, False if the key is not present in the hash map or if the key associated value does not match expected.
  bool remove(Key key, Value expected_value);
#endif  // BASIC_HASH_MAP

  // #ifdef MEM_CHECK
  int getMem() {
    return (sizeof(GPtr) * this->this_head_length +                       // head
            sizeof(GPtr) * this->array_length * this->array_mem.size() +  // array
            sizeof(DataNode) * data_mem.size());                          // data
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
  /// @brief Expand the map when there is a hash collision.
  /// @param local Address need to expand.
  /// @param pos Displacement at local.
  /// @param right Right-shifted bits
  /// @return Address to expanded part.
  GPtr expandMap(GPtr local, int pos, int right);
  /// @brief Mark a node as hazard.
  /// @param node Pointer to node.
  void watch(GPtr node);
  /// @brief Free node.
  /// @param node_to_free Node to free.
  void free(GPtr node_to_free);
  /// @brief Ensure momory is not freed while another process is using it.
  /// @param node_to_free Node to free
  void safeFreeNode(GPtr node_to_free);
  /// @brief Allocate data node, reuse memory before alloc new.
  /// @param value Value.
  /// @param hash_value Hash value.
  /// @return Pointer to this node.
  GPtr allocateNode(Value value, HashValue hash_value);
  /// @brief Scan deleted node for reuse.
  void scan();
  /// @brief Get pointer at local[position].
  /// @param local Address.
  /// @param position Displacement.
  /// @return Pointer.
  GPtr getNode(GPtr local, int position);
  /// @brief Set pointer at local[position].
  /// @param local Address.
  /// @param position Displacement.
  /// @param node New pointer.
  void setNode(GPtr local, int position, GPtr node);
  /// @brief Get data node.
  /// @param node Address.
  /// @return Data node.
  DataNode getData(GPtr node);
  /// @brief Set data node.
  /// @param node Address.
  /// @param value New value.
  /// @param hash_value New hash value.
  void setData(GPtr node, Value value, HashValue hash_value);
  /// @brief Check if node is marked.
  /// @param node Address.
  /// @return True if marked, False otherwise.
  bool isMarked(GPtr node);
  /// @brief Check if node is an array node.
  /// @param node Address.
  /// @return True if array node, False otherwise.
  bool isArrayNode(GPtr node);
  /// @brief Mark data node at local[position]
  /// @param local Address.
  /// @param pos Displacement.
  /// @return Pointer to that node.
  GPtr markDataNode(GPtr local, int position);
  GPtr unmark(GPtr node);

 private:
  /// @brief Get hazard pointer.
  /// @param rank Rank.
  /// @return Pointer.
  GPtr getHPtr(int rank);
  /// @brief Make global pointer by rank(14 bit) - disp(48 bit) - tag(2 bit)
  /// @param rank Rank
  /// @param disp Displacement
  /// @param tag Tag
  /// @return Pointer.
  GPtr makeGPtr(int rank, MPI_Aint disp, int tag);
  /// @brief Get rank of position in global head.
  /// @param position Position.
  /// @return Rank.
  int getHeadRank(int position);
  /// @brief Get real displacement in process has position of global head.
  /// @param position Position.
  /// @return Displacement.
  MPI_Aint getHeadDisp(int position);
  /// @brief Get rank of global pointer.
  /// @param node Pointer.
  /// @return Rank.
  int getRank(GPtr node);
  /// @brief Get displacement of global pointer.
  /// @param node Pointer.
  /// @return Displacement.
  MPI_Aint getDisp(GPtr node);
  /// @brief Compare and Swap.
  /// @param local Address.
  /// @param position Displacement at local.
  /// @param old_value Expected value.
  /// @param new_value New value.
  /// @return Value at local[position] at the beggining.
  GPtr CAS(GPtr local, int position, GPtr old_value, GPtr new_value);

 private:
  int nprocs;
  int myrank;
  /// @brief Size of Key in bit
  int key_size;
  /// @brief Log2 of head_length
  int head_pow;
  /// @brief Length of Head Array
  int head_length;
  /// @brief Log2 of array_length
  int array_pow;
  /// @brief Length of all Array Node
  int array_length;
  int this_head_length;
  int max_fail_count = 10;
  std::size_t max_dlist_size = 10;
  HashFunctor hash_functor;

#ifdef COMM_CHECK
  bool isIntra(int source_rank, int target_rank) {
    return (source_rank / 8) == (target_rank / 8);
  }
  int intra_comm_count = 0;
  int inter_comm_count = 0;
#endif  // COMM_CHECK

 private:
  MPI_Comm comm;
  MPI_Win head;
  MPI_Win array;
  MPI_Win data;

 private:
  std::list<GPtr*> array_mem;
  std::list<DataNode*> data_mem;
  MPI_Win hp_win;
  GPtr* hp;
  std::list<GPtr> plist;
  std::list<GPtr> rlist;
  std::list<GPtr> dlist;
  GPtr* head_arr;

  class DataNode {
    using HashValue = WaitFreeHashMap::HashValue;

   public:
    DataNode() : hash_value(), value() {};
    DataNode(HashValue hash_value, Value value) : hash_value(hash_value), value(value) {};

   public:
    HashValue hash_value;
    Value value;
  };
};

template <typename Key, typename Value, typename HashFunctor>
inline WaitFreeHashMap<Key, Value, HashFunctor>::WaitFreeHashMap(MPI_Comm comm) : comm(comm), array_mem(), data_mem(), plist(), rlist(), dlist() {
  // Get number of processes and rank
  MPI_Comm_size(comm, &this->nprocs);
  MPI_Comm_rank(comm, &this->myrank);
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]" << ": constructing..." << std::endl;
#endif  // DEBUG
  // Calculate constants
  this->key_size = sizeof(Key) * 8;
  this->array_pow = sqrt(key_size);
  int max_depth = this->key_size / this->array_pow;
  this->head_pow = this->key_size - this->array_pow * (max_depth - 1);
  this->head_length = 1 << this->head_pow;
  this->array_length = 1 << this->array_pow;
  this->this_head_length = this->head_length / this->nprocs;
  if (myrank + 1 == nprocs)
    this->this_head_length += this->head_length % this->nprocs;
#ifdef DEBUG
  if (this->myrank == 0) {
    std::cout << "[[]]: key size    : " << this->key_size << std::endl;
    std::cout << "[[]]: head length : " << this->head_length << std::endl;
    std::cout << "[[]]: head pow    : " << this->head_pow << std::endl;
    std::cout << "[[]]: array length: " << this->array_length << std::endl;
    std::cout << "[[]]: array pow   : " << this->array_pow << std::endl;
  }
#endif  // DEBUG
  // Create windows
  this->head_arr = new GPtr[this->this_head_length]();
  MPI_Win_create(this->head_arr, this->this_head_length * sizeof(GPtr), sizeof(GPtr), MPI_INFO_NULL, comm, &this->head);
  MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &this->array);
  MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &this->data);
  this->hp = new GPtr();
  MPI_Win_create(this->hp, sizeof(GPtr), sizeof(GPtr), MPI_INFO_NULL, comm, &this->hp_win);
  MPI_Win_fence(0, this->head);
  MPI_Win_fence(0, this->array);
  MPI_Win_fence(0, this->data);
  MPI_Win_fence(0, this->hp_win);
  MPI_Win_lock_all(0, this->head);
  MPI_Win_lock_all(0, this->array);
  MPI_Win_lock_all(0, this->data);
  MPI_Win_lock_all(0, this->hp_win);
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]" << ": construct successfully" << std::endl;
#endif  // DEBUG
}
template <typename Key, typename Value, typename HashFunctor>
inline WaitFreeHashMap<Key, Value, HashFunctor>::~WaitFreeHashMap() {
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]" << ": destruct" << std::endl;
#endif  // DEBUG
  int flag;
  delete[] this->head_arr;
  for (std::list<GPtr*>::iterator it = array_mem.begin(); it != array_mem.end(); ++it) {
    delete[] *it;
  }
  for (typename std::list<DataNode*>::iterator it = data_mem.begin(); it != data_mem.end(); ++it) {
    delete *it;
  }
  delete this->hp;
  MPI_Finalized(&flag);
  if (!flag) {
    MPI_Win_unlock_all(this->head);
    MPI_Win_unlock_all(this->array);
    MPI_Win_unlock_all(this->data);
    MPI_Win_unlock_all(this->hp_win);
    MPI_Win_free(&this->head);
    MPI_Win_free(&this->array);
    MPI_Win_free(&this->data);
    MPI_Win_free(&this->hp_win);
  }
}
template <typename Key, typename Value, typename HashFunctor>
bool WaitFreeHashMap<Key, Value, HashFunctor>::insert(Key key, Value value) {
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]" << ": inserting<" << key << "," << value << ">..." << std::endl;
#endif  // DEBUG
  HashValue hash_value = hash_functor(key);
  HashValue hash_value_temp = hash_value;

  HashValue head_bits;
  head_bits.set();
  HashValue array_bits = head_bits >> (this->key_size - this->array_pow);
  head_bits >>= (this->key_size - this->head_pow);

  GPtr local = null_gptr, node = null_gptr, insert_this = null_gptr;
  int position = 0;

  for (int right = 0; right < this->key_size;) {
    position = 0;
    int fail_count = 0;
    if (local == null_gptr) {
      position = (hash_value_temp & head_bits).to_ulong();
      hash_value_temp >>= this->head_pow;
      right += head_pow;
    } else {
      position = (hash_value_temp & array_bits).to_ulong();
      hash_value_temp >>= this->array_pow;
      right += array_pow;
    }
    node = this->getNode(local, position);
    while (true) {
      if (fail_count > this->max_fail_count)
        node = markDataNode(local, position);
      if (node == null_gptr) {
        insert_this = this->allocateNode(value, hash_value);
        if ((node = this->CAS(local, position, null_gptr, insert_this)) == null_gptr) {
          this->watch(null_gptr);
          return true;
        } else
          this->free(insert_this);
      }
      if (this->isMarked(node))
        node = this->expandMap(local, position, right);
      if (this->isArrayNode(node)) {
        local = node;
        break;
      } else {
        this->watch(node);
        GPtr node2 = getNode(local, position);
        if (node != node2) {
          ++fail_count;
          node = node2;
          continue;
        } else if (this->getData(node).hash_value == hash_value) {
          this->watch(null_gptr);
          return false;
        } else {
          node = this->expandMap(local, position, right);
          if (this->isArrayNode(node)) {
            local = node;
            break;
          } else
            ++fail_count;
        }
      }
    }
  }
  this->free(insert_this);
  this->watch(null_gptr);
  position = (hash_value_temp & array_bits).to_ulong();
  GPtr curr_node = this->getNode(local, position);
  if (curr_node == null_gptr) {
    insert_this = this->allocateNode(value, hash_value);
    if ((node = this->CAS(local, position, null_gptr, insert_this)) == null_gptr) {
      return true;
    } else {
      this->free(insert_this);
      return false;
    }
  } else
    return false;
}
#ifdef BASIC_HASH_MAP
#else

template <typename Key, typename Value, typename HashFunctor>
inline bool WaitFreeHashMap<Key, Value, HashFunctor>::update(Key key, Value expected_value, Value new_value) {
  HashValue hash_value = hash_functor(key);
  HashValue hash_value_temp = hash_value;

  HashValue head_bits;
  head_bits.set();
  HashValue array_bits = head_bits >> (this->key_size - this->array_pow);
  head_bits >>= (this->key_size - this->head_pow);

  GPtr local = null_gptr, node = null_gptr, insert_this = null_gptr;
  int position = 0;

  bool result = false;
  int right = 0;

  for (right = 0; right < this->key_size;) {
    position = 0;
    int fail_count = 0;
    if (local == null_gptr) {
      position = (hash_value_temp & head_bits).to_ulong();
      hash_value_temp >>= this->head_pow;
      right += head_pow;
    } else {
      position = (hash_value_temp & array_bits).to_ulong();
      hash_value_temp >>= this->array_pow;
      right += array_pow;
    }
    node = this->getNode(local, position);

    if (this->isArrayNode(node))
      local = node;
    else if (this->isMarked(node))
      local = this->expandMap(local, position, right);
    else if (node == null_gptr)
      break;
    else {
      this->watch(node);
      if (node != this->getNode(local, position)) {
        fail_count = 0;
        while (node != this->getNode(local, position)) {
          node = this->getNode(local, position);
          this->watch(node);
          ++fail_count;
          if (fail_count > this->max_fail_count) {
            this->markDataNode(local, position);
            node = this->expandMap(local, position, right);
            break;
          }
        }
        if (this->isArrayNode(node)) {
          local = node;
          continue;
        } else if (this->isMarked(node)) {
          local = this->expandMap(local, position, right);
          continue;
        } else if (node == null_gptr)
          break;
      }
      DataNode data_node = this->getData(node);
      if (data_node.hash_value == hash_value) {
        if (data_node.value != expected_value)
          break;
        GPtr node2 = null_gptr;
        insert_this = this->allocateNode(new_value, hash_value);
        if ((node2 = this->CAS(local, position, node, insert_this)) == node) {
          this->safeFreeNode(node);
          result = true;
          break;
        } else {
          this->free(insert_this);
          if (this->isArrayNode(node2))
            local = node2;
          else if ((this->isMarked(node2) ^ this->unmark(node2)) == node)
            local = this->expandMap(local, position, right);
          else
            break;
        }
      } else
        break;
    }
  }
  if (right >= this->key_size) {
    position = (hash_value_temp & array_bits).to_ulong();
    GPtr curr_node = this->getNode(local, position);
    if (curr_node != null_gptr) {
      if (this->getData(curr_node).value == expected_value) {
        insert_this = this->allocateNode(new_value, hash_value);
        result = (this->CAS(local, position, curr_node, insert_this) == curr_node);
        if (result == true)
          this->safeFreeNode(curr_node);
        else
          this->free(insert_this);
      } else {
        result = false;
      }
    }
  }
  this->watch(null_gptr);
  return result;
}
#endif  // BASIC_HASH_MAP
#ifdef BASIC_HASH_MAP
template <typename Key, typename Value, typename HashFunctor>
bool WaitFreeHashMap<Key, Value, HashFunctor>::get(Key key, Value& value) {
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]" << ": getting<" << key << ">..." << std::endl;
#endif  // DEBUG
  /*TODO: return null_gptr if no key match*/
  HashValue hash_value = hash_functor(key);
  HashValue hash_value_temp = hash_value;

  HashValue head_bits;
  head_bits.set();
  HashValue array_bits = head_bits >> (this->key_size - this->array_pow);
  head_bits >>= (this->key_size - this->head_pow);

  GPtr local = null_gptr, node = null_gptr;
  int position = 0;

  bool result = false;
  int right = 0;

  for (right = 0; right < this->key_size;) {
    position = 0;
    int fail_count = 0;
    if (local == null_gptr) {
      position = (hash_value_temp & head_bits).to_ulong();
      hash_value_temp >>= this->head_pow;
      right += head_pow;
    } else {
      position = (hash_value_temp & array_bits).to_ulong();
      hash_value_temp >>= this->array_pow;
      right += array_pow;
    }
    node = this->getNode(local, position);

    if (this->isArrayNode(node))
      local = node;
    else if (node == null_gptr)
      break;
    else {
      this->watch(node);
      if (node != this->getNode(local, position)) {
        fail_count = 0;
        while (node != this->getNode(local, position)) {
          node = this->getNode(local, position);
          this->watch(node);
          ++fail_count;
          if (fail_count > this->max_fail_count) {
            this->markDataNode(local, position);
            local = this->expandMap(local, position, right);
            break;
          }
        }
        if (this->isArrayNode(node)) {
          local = node;
          continue;
        } else if (this->isMarked(node)) {
          local = this->expandMap(local, position, right);
          continue;
        } else if (node == null_gptr)
          break;
      }
      if (this->getData(node).hash_value == hash_value) /*hash_value vs hash_value_begin*/ {
        value = ((this->getData(node)).value);
        result = true;
      }
      break;
    }
  }
  if (right >= this->key_size) {
    position = (hash_value_temp & array_bits).to_ulong();
    GPtr curr_node = this->getNode(local, position);
    if (curr_node != null_gptr) {
      value = ((this->getData(curr_node)).value);
      result = true;
    }
  }
  this->watch(null_gptr);
  return result;
}
#else
template <typename Key, typename Value, typename HashFunctor>
Value WaitFreeHashMap<Key, Value, HashFunctor>::get(Key key) {
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]" << ": getting<" << key << ">..." << std::endl;
#endif  // DEBUG
  /*TODO: return null_gptr if no key match*/
  HashValue hash_value = hash_functor(key);
  HashValue hash_value_temp = hash_value;

  HashValue head_bits;
  head_bits.set();
  HashValue array_bits = head_bits >> (this->key_size - this->array_pow);
  head_bits >>= (this->key_size - this->head_pow);

  GPtr local = null_gptr, node = null_gptr;
  int position = 0;

  Value result{};
  int right = 0;

  for (right = 0; right < this->key_size;) {
    position = 0;
    int fail_count = 0;
    if (local == null_gptr) {
      position = (hash_value_temp & head_bits).to_ulong();
      hash_value_temp >>= this->head_pow;
      right += head_pow;
    } else {
      position = (hash_value_temp & array_bits).to_ulong();
      hash_value_temp >>= this->array_pow;
      right += array_pow;
    }
    node = this->getNode(local, position);

    if (this->isArrayNode(node))
      local = node;
    else if (node == null_gptr)
      break;
    else {
      this->watch(node);
      if (node != this->getNode(local, position)) {
        fail_count = 0;
        while (node != this->getNode(local, position)) {
          node = this->getNode(local, position);
          this->watch(node);
          ++fail_count;
          if (fail_count > this->max_fail_count) {
            this->markDataNode(local, position);
            local = this->expandMap(local, position, right);
            break;
          }
        }
        if (this->isArrayNode(node)) {
          local = node;
          continue;
        } else if (this->isMarked(node)) {
          local = this->expandMap(local, position, right);
          continue;
        } else if (node == null_gptr)
          break;
      }
      if (this->getData(node).hash_value == hash_value) /*hash_value vs hash_value_begin*/ {
        result = ((this->getData(node)).value);
      }
      break;
    }
  }
  if (right >= this->key_size) {
    position = (hash_value_temp & array_bits).to_ulong();
    GPtr curr_node = this->getNode(local, position);
    if (curr_node != null_gptr)
      result = ((this->getData(curr_node)).value);
  }
  this->watch(null_gptr);
  return result;
}
#endif  // BASIC_HASH_MAP
#ifdef BASIC_HASH_MAP
template <typename Key, typename Value, typename HashFunctor>
bool WaitFreeHashMap<Key, Value, HashFunctor>::remove(Key key) {
  HashValue hash_value = hash_functor(key);
  HashValue hash_value_temp = hash_value;

  HashValue head_bits;
  head_bits.set();
  HashValue array_bits = head_bits >> (this->key_size - this->array_pow);
  head_bits >>= (this->key_size - this->head_pow);

  GPtr local = null_gptr, node = null_gptr;
  int position = 0;

  bool result = false;
  int right = 0;

  for (right = 0; right < this->key_size;) {
    position = 0;
    int fail_count = 0;
    if (local == null_gptr) {
      position = (hash_value_temp & head_bits).to_ulong();
      hash_value_temp >>= this->head_pow;
      right += head_pow;
    } else {
      position = (hash_value_temp & array_bits).to_ulong();
      hash_value_temp >>= this->array_pow;
      right += array_pow;
    }
    node = this->getNode(local, position);

    if (this->isArrayNode(node))
      local = node;
    else if (this->isMarked(node))
      local = this->expandMap(local, position, right);
    else if (node == null_gptr)
      break;
    else {
      this->watch(node);
      if (node != this->getNode(local, position)) {
        fail_count = 0;
        while (node != this->getNode(local, position)) {
          node = this->getNode(local, position);
          this->watch(node);
          ++fail_count;
          if (fail_count > this->max_fail_count) {
            this->markDataNode(local, position);
            node = this->expandMap(local, position, right);
            break;
          }
        }
        if (this->isArrayNode(node)) {
          local = node;
          continue;
        } else if (this->isMarked(node)) {
          local = this->expandMap(local, position, right);
          continue;
        } else if (node == null_gptr)
          break;
      }
      DataNode data_node = this->getData(node);
      if (data_node.hash_value == hash_value) {
        GPtr node2 = null_gptr;
        if ((node2 = this->CAS(local, position, node, null_gptr)) == node) {
          this->safeFreeNode(node);
          result = true;
          break;
        } else {
          if (this->isArrayNode(node2))
            local = node2;
          else if ((this->isMarked(node2) ^ this->unmark(node2)) == node)
            local = this->expandMap(local, position, right);
          else
            break;
        }
      } else
        break;
    }
  }
  if (right >= this->key_size) {
    position = (hash_value_temp & array_bits).to_ulong();
    GPtr curr_node = this->getNode(local, position);
    if (curr_node != null_gptr) {
      result = (this->CAS(local, position, curr_node, null_gptr) == curr_node);
      if (result == true)
        this->safeFreeNode(curr_node);
    }
  }
  this->watch(null_gptr);
  return result;
}
#else
template <typename Key, typename Value, typename HashFunctor>
bool WaitFreeHashMap<Key, Value, HashFunctor>::remove(Key key, Value expected_value) {
  HashValue hash_value = hash_functor(key);
  HashValue hash_value_temp = hash_value;

  HashValue head_bits;
  head_bits.set();
  HashValue array_bits = head_bits >> (this->key_size - this->array_pow);
  head_bits >>= (this->key_size - this->head_pow);

  GPtr local = null_gptr, node = null_gptr;
  int position = 0;

  bool result = false;
  int right = 0;

  for (right = 0; right < this->key_size;) {
    position = 0;
    int fail_count = 0;
    if (local == null_gptr) {
      position = (hash_value_temp & head_bits).to_ulong();
      hash_value_temp >>= this->head_pow;
      right += head_pow;
    } else {
      position = (hash_value_temp & array_bits).to_ulong();
      hash_value_temp >>= this->array_pow;
      right += array_pow;
    }
    node = this->getNode(local, position);

    if (this->isArrayNode(node))
      local = node;
    else if (this->isMarked(node))
      local = this->expandMap(local, position, right);
    else if (node == null_gptr)
      break;
    else {
      this->watch(node);
      if (node != this->getNode(local, position)) {
        fail_count = 0;
        while (node != this->getNode(local, position)) {
          node = this->getNode(local, position);
          this->watch(node);
          ++fail_count;
          if (fail_count > this->max_fail_count) {
            this->markDataNode(local, position);
            node = this->expandMap(local, position, right);
            break;
          }
        }
        if (this->isArrayNode(node)) {
          local = node;
          continue;
        } else if (this->isMarked(node)) {
          local = this->expandMap(local, position, right);
          continue;
        } else if (node == null_gptr)
          break;
      }
      DataNode data_node = this->getData(node);
      if (data_node.hash_value == hash_value) {
        if (data_node.value != expected_value)
          break;
        GPtr node2 = null_gptr;
        if ((node2 = this->CAS(local, position, node, null_gptr)) == node) {
          this->safeFreeNode(node);
          result = true;
          break;
        } else {
          if (this->isArrayNode(node2))
            local = node2;
          else if ((this->isMarked(node2) ^ this->unmark(node2)) == node)
            local = this->expandMap(local, position, right);
          else
            break;
        }
      } else
        break;
    }
  }
  if (right >= this->key_size) {
    position = (hash_value_temp & array_bits).to_ulong();
    GPtr curr_node = this->getNode(local, position);
    if (curr_node != null_gptr) {
      if (this->getData(curr_node).value == expected_value) {
        result = (this->CAS(local, position, curr_node, null_gptr) == curr_node);
        if (result == true)
          this->safeFreeNode(curr_node);
      } else {
        result = false;
      }
    }
  }
  this->watch(null_gptr);
  return result;
}
#endif  // BASIC_HASH_MAP
template <typename Key, typename Value, typename HashFunctor>
inline typename WaitFreeHashMap<Key, Value, HashFunctor>::GPtr WaitFreeHashMap<Key, Value, HashFunctor>::expandMap(GPtr local, int position, int right) {
#ifdef DEBUG
  std::cout << this->myrank << ": expanding..." << std::endl;
#endif  // DEBUG
  HashValue head_bits;
  head_bits.set();
  HashValue array_bits = head_bits >> (this->key_size - this->array_pow);
  head_bits >>= (this->key_size - this->head_pow);

  GPtr node = this->getNode(local, position);
  GPtr node2;
  this->watch(node);
  if (this->isArrayNode(node))
    return node;
  if (node != (node2 = this->getNode(local, position)))
    return node2;

  GPtr* arr = new GPtr[this->array_length]();
  MPI_Win_attach(this->array, arr, this->array_length * sizeof(GPtr));
  MPI_Aint addr;
  MPI_Get_address(arr, &addr);
  GPtr a_node = this->makeGPtr(this->myrank, addr, 2);

  HashValue hash_value_temp = this->getData(node).hash_value;
  hash_value_temp >>= right;

  int new_position = (hash_value_temp & array_bits).to_ulong();
  this->setNode(a_node, new_position, node);
  if ((node2 = this->CAS(local, position, node, a_node)) == node) {
    this->array_mem.push_back(arr);
    return a_node;
  } else {
    this->setNode(a_node, new_position, null_gptr);
    MPI_Win_detach(this->array, arr);
    delete[] arr;
    return node2;
  }
}
template <typename Key, typename Value, typename HashFunctor>
inline void WaitFreeHashMap<Key, Value, HashFunctor>::watch(GPtr node) {
#ifdef COMM_CHECK
  if (isIntra(this->myrank, this->myrank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  MPI_Accumulate(&node, sizeof(GPtr), MPI_CHAR, this->myrank, 0, sizeof(GPtr), MPI_CHAR, MPI_REPLACE, this->hp_win);
  MPI_Win_flush(this->myrank, this->hp_win);
}
template <typename Key, typename Value, typename HashFunctor>
inline void WaitFreeHashMap<Key, Value, HashFunctor>::free(GPtr node_to_free) {
  this->plist.push_back(node_to_free);
}
template <typename Key, typename Value, typename HashFunctor>
inline void WaitFreeHashMap<Key, Value, HashFunctor>::safeFreeNode(GPtr node_to_free) {
  this->dlist.push_back(node_to_free);
  if (this->dlist.size() >= this->max_dlist_size)
    this->scan();
}
template <typename Key, typename Value, typename HashFunctor>
inline typename WaitFreeHashMap<Key, Value, HashFunctor>::GPtr WaitFreeHashMap<Key, Value, HashFunctor>::allocateNode(Value value, HashValue hash_value) {
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
    DataNode* data_node;
    try {
      data_node = new DataNode();
    } catch (const std::bad_alloc& e) {
      std::cout << "   ------ [[Rank = " << this->myrank << "]: bad alloc: " << e.what() << std::endl;
      while (true) {
        sleep(10);
      }
    }
    this->data_mem.push_back(data_node);
    MPI_Win_attach(this->data, data_node, sizeof(DataNode));
    MPI_Get_address(data_node, &addr);
    result = this->makeGPtr(this->myrank, addr, 0);
  }
  this->setData(result, value, hash_value);
#ifdef DEBUG
  std::cout << this->myrank << ":alloc<" << hash_value << "," << value << ">:" << result << std::endl;
#endif  // DEBUG
  return result;
}
template <typename Key, typename Value, typename HashFunctor>
inline void WaitFreeHashMap<Key, Value, HashFunctor>::scan() {
  std::list<GPtr> list_temp;
  std::list<GPtr> dlist_temp;
  GPtr hp;
  for (int i = 0; i < this->nprocs; ++i) {
    hp = this->getHPtr(i);
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
template <typename Key, typename Value, typename HashFunctor>
inline typename WaitFreeHashMap<Key, Value, HashFunctor>::GPtr WaitFreeHashMap<Key, Value, HashFunctor>::getNode(GPtr local, int pos) {
  GPtr result = 2;
  if (local == null_gptr) {
    int rank = this->getHeadRank(pos);
#ifdef COMM_CHECK
    if (isIntra(this->myrank, rank) == true)
      ++(this->intra_comm_count);
    else
      ++(this->inter_comm_count);
#endif  // COMM_CHECK
    MPI_Aint disp = this->getHeadDisp(pos);
    MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(GPtr), MPI_CHAR, rank, disp, sizeof(GPtr), MPI_CHAR, MPI_NO_OP, this->head);
    MPI_Win_flush(rank, this->head);
  } else {
    int rank = this->getRank(local);
#ifdef COMM_CHECK
    if (isIntra(this->myrank, rank) == true)
      ++(this->intra_comm_count);
    else
      ++(this->inter_comm_count);
#endif  // COMM_CHECK
    MPI_Aint disp = this->getDisp(local) + pos * sizeof(GPtr);
    MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(GPtr), MPI_CHAR, rank, disp, sizeof(GPtr), MPI_CHAR, MPI_NO_OP, this->array);
    MPI_Win_flush(rank, this->array);
  }
#ifdef DEBUG
  std::cout << "///" << this->myrank << ": getnode: " << local << "-" << pos << ":" << result << std::endl;
#endif  // DEBUG
  return result;
}
template <typename Key, typename Value, typename HashFunctor>
inline void WaitFreeHashMap<Key, Value, HashFunctor>::setNode(GPtr local, int position, GPtr node) {
#ifdef DEBUG
  std::cout << "///" << this->myrank << ": setnode: " << local << "-" << position << ":" << node << std::endl;
#endif  // DEBUG
  if (local == null_gptr) {
    int rank = this->getHeadRank(position);
#ifdef COMM_CHECK
    if (isIntra(this->myrank, rank) == true)
      ++(this->intra_comm_count);
    else
      ++(this->inter_comm_count);
#endif  // COMM_CHECK
    MPI_Aint disp = this->getHeadDisp(position);
    MPI_Accumulate(&node, 1, MPI_UINT64_T, rank, disp, 1, MPI_UINT64_T, MPI_REPLACE, this->head);
    MPI_Win_flush(rank, this->head);
  } else {
    int rank = this->getRank(local);
#ifdef COMM_CHECK
    if (isIntra(this->myrank, rank) == true)
      ++(this->intra_comm_count);
    else
      ++(this->inter_comm_count);
#endif  // COMM_CHECK
    MPI_Aint disp = this->getDisp(local) + position * sizeof(GPtr);
    MPI_Accumulate(&node, 1, MPI_UINT64_T, rank, disp, 1, MPI_UINT64_T, MPI_REPLACE, this->array);
    MPI_Win_flush(rank, this->array);
  }
#ifdef DEBUG
  std::cout << "///" << this->myrank << ": setnode(completed): " << local << "-" << position << ":" << node << std::endl;
#endif  // DEBUG
}
template <typename Key, typename Value, typename HashFunctor>
inline typename WaitFreeHashMap<Key, Value, HashFunctor>::DataNode WaitFreeHashMap<Key, Value, HashFunctor>::getData(GPtr node) {
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
#ifdef DEBUG
  std::cout << this->myrank << ": get data at: " << node << ":<" << result.hash_value << "," << result.value << ">" << std::endl;
#endif  // DEBUG
  return result;
}
template <typename Key, typename Value, typename HashFunctor>
inline void WaitFreeHashMap<Key, Value, HashFunctor>::setData(GPtr node, Value value, HashValue hash_value) {
  DataNode data_node(hash_value, value);
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
#ifdef DEBUG
  std::cout << this->myrank << ": set data at: " << node << ":<" << value << "," << hash_value << ">" << std::endl;
#endif  // DEBUG
}
template <typename Key, typename Value, typename HashFunctor>
inline bool WaitFreeHashMap<Key, Value, HashFunctor>::isMarked(GPtr node) {
#ifdef DEBUG
  std::cout << this->myrank << ":check mark:" << node << ":" << ((node & 1)) << std::endl;
#endif  // DEBUG
  return (node & 1);
}
template <typename Key, typename Value, typename HashFunctor>
inline bool WaitFreeHashMap<Key, Value, HashFunctor>::isArrayNode(GPtr node) {
#ifdef DEBUG
  std::cout << this->myrank << ":check array:" << node << ":" << ((node & 2)) << std::endl;
#endif  // DEBUG
  return (node & 2);
}
template <typename Key, typename Value, typename HashFunctor>
inline typename WaitFreeHashMap<Key, Value, HashFunctor>::GPtr WaitFreeHashMap<Key, Value, HashFunctor>::markDataNode(GPtr local, int pos) {
#ifdef DEBUG
  std::cout << this->myrank << "mark data: <" << local << "," << pos << ">" << std::endl;
#endif  // DEBUG
  GPtr m = 1;
  if (local == null_gptr) {
    int rank = this->getHeadRank(pos);
#ifdef COMM_CHECK
    if (isIntra(this->myrank, rank) == true)
      ++(this->intra_comm_count);
    else
      ++(this->inter_comm_count);
#endif  // COMM_CHECK
    MPI_Aint disp = this->getHeadDisp(pos);
    MPI_Fetch_and_op(&m, NULL, MPI_UINT64_T, rank, disp, MPI_BOR, this->head);
    MPI_Win_flush(rank, this->head);
  } else {
    int rank = this->getRank(local);
#ifdef COMM_CHECK
    if (isIntra(this->myrank, rank) == true)
      ++(this->intra_comm_count);
    else
      ++(this->inter_comm_count);
#endif  // COMM_CHECK
    MPI_Aint disp = this->getDisp(local) + pos * sizeof(GPtr);
    MPI_Fetch_and_op(&m, NULL, MPI_UINT64_T, rank, disp, MPI_BOR, this->array);
    MPI_Win_flush(rank, this->array);
  }
  return this->getNode(local, pos);
}
template <typename Key, typename Value, typename HashFunctor>
inline typename WaitFreeHashMap<Key, Value, HashFunctor>::GPtr WaitFreeHashMap<Key, Value, HashFunctor>::unmark(GPtr node) {
  return (node | 1);
}
template <typename Key, typename Value, typename HashFunctor>
inline typename WaitFreeHashMap<Key, Value, HashFunctor>::GPtr WaitFreeHashMap<Key, Value, HashFunctor>::getHPtr(int rank) {
#ifdef COMM_CHECK
  if (isIntra(this->myrank, rank) == true)
    ++(this->intra_comm_count);
  else
    ++(this->inter_comm_count);
#endif  // COMM_CHECK
  GPtr result = 2;
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(GPtr), MPI_CHAR, rank, 0, sizeof(GPtr), MPI_CHAR, MPI_NO_OP, this->hp_win);
  MPI_Win_flush(rank, this->hp_win);
  return result;
}
template <typename Key, typename Value, typename HashFunctor>
inline typename WaitFreeHashMap<Key, Value, HashFunctor>::GPtr WaitFreeHashMap<Key, Value, HashFunctor>::makeGPtr(int rank, MPI_Aint disp, int tag) {
  return ((uint64_t)rank << 50) | (disp << 2) | tag;
}
template <typename Key, typename Value, typename HashFunctor>
inline int WaitFreeHashMap<Key, Value, HashFunctor>::getHeadRank(int position) {
  return std::min(position / (this->head_length / this->nprocs), this->nprocs - 1);
}
template <typename Key, typename Value, typename HashFunctor>
inline MPI_Aint WaitFreeHashMap<Key, Value, HashFunctor>::getHeadDisp(int position) {
  return position - this->getHeadRank(position) * (this->head_length / this->nprocs);
}
template <typename Key, typename Value, typename HashFunctor>
inline int WaitFreeHashMap<Key, Value, HashFunctor>::getRank(GPtr node) {
  return (node >> 50);
}
template <typename Key, typename Value, typename HashFunctor>
inline MPI_Aint WaitFreeHashMap<Key, Value, HashFunctor>::getDisp(GPtr node) {
  return ((node << 14) >> 16);
}
template <typename Key, typename Value, typename HashFunctor>
inline typename WaitFreeHashMap<Key, Value, HashFunctor>::GPtr
WaitFreeHashMap<Key, Value, HashFunctor>::CAS(GPtr local, int position, GPtr old_value, GPtr new_value) {
#ifdef DEBUG
  std::cout << this->myrank << ": cas<" << local << "," << position << ">/new value: " << new_value << std::endl;
#endif  // DEBUG
  GPtr result;
  if (local == null_gptr) {
    int rank = this->getHeadRank(position);
#ifdef COMM_CHECK
    if (isIntra(this->myrank, rank) == true)
      ++(this->intra_comm_count);
    else
      ++(this->inter_comm_count);
#endif  // COMM_CHECK
    MPI_Aint disp = this->getHeadDisp(position);
    MPI_Compare_and_swap(&new_value, &old_value, &result, MPI_UINT64_T, rank, disp, this->head);
    MPI_Win_flush(rank, this->head);
  } else {
    int rank = this->getRank(local);
#ifdef COMM_CHECK
    if (isIntra(this->myrank, rank) == true)
      ++(this->intra_comm_count);
    else
      ++(this->inter_comm_count);
#endif  // COMM_CHECK
    MPI_Aint disp = this->getDisp(local) + position * sizeof(GPtr);
    MPI_Compare_and_swap(&new_value, &old_value, &result, MPI_UINT64_T, rank, disp, this->array);
    MPI_Win_flush(rank, this->array);
  }
  return result;
}

}  // namespace ngu

#endif  // DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_WAIT_FREE_HASH_MAP_H