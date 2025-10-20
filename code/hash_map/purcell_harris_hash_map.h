#ifndef DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_PURCELL_HARRIS_HASH_MAP_H
#define DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_PURCELL_HARRIS_HASH_MAP_H

#include <mpi.h>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <functional>
#include <iostream>
#include <list>
#include <random>

namespace ngu {
uint64_t splitmix64_2a(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x = x ^ (x >> 31);
  return x;
}
// template <typename Key>
// class Compare {
//  public:
//   bool operator()(Key key1, Key key2) const {
//     return key1 < key2;
//   };
// };

template <typename Key>
class Hash4 {
 public:
  uint64_t operator()(Key key) const {
    std::hash<Key> h;
    return splitmix64_2a(h(key) + 12345);
  };
};

template <typename Key, typename Value, typename HashFunctor = Hash4<Key>>
class PurcellHarrisHashMap {
  /// @brief bound(63 bit) - scanning(1 bit)
  using BoundType = uint64_t;
  /// @brief version(61 bit) - state(3 bit) [0:empty,1:busy,2:collied,3:visible,4:inserting,5:member]
  using VersionStateType = uint64_t;

  class DataNode;

 public:
  PurcellHarrisHashMap(MPI_Comm comm = MPI_COMM_WORLD, int table_length_each = 1000);
  PurcellHarrisHashMap(const PurcellHarrisHashMap&) = delete;
  PurcellHarrisHashMap& operator=(const PurcellHarrisHashMap&) = delete;
  ~PurcellHarrisHashMap();
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
    return (sizeof(BoundType) * this->table_length_each +         // bound
            sizeof(VersionStateType) * this->table_length_each +  // version state
            sizeof(DataNode) * this->table_length_each);          // data
  }
  // #endif  // MEM_CHECK

 private:
  bool assist(Key key, int position, int index, int version);
  void conditionallyRaiseBound(int position, int index);
  void conditionallyLowerBound(int position, int index);
  int getBucket(int position, int index);
  bool doesBucketContainCollision(int position, int index);

 private:
  int getBound(int position);
  int getBoundFromProbeBound(BoundType bound);
  bool isScanning(BoundType bound);
  BoundType boundUnscan(BoundType bound);
  BoundType makeProbeBound(int bound, bool scanning);

  int getVersion(VersionStateType version_state);
  int getState(VersionStateType version_state);
  VersionStateType makeVersionState(int version, int state);

  bool isVisible(int state);
  bool isInserting(int state);
  bool isMember(int state);

 private:
  int getRank(int position, int index = 0);
  int getDisp(int position, int index = 0);
  BoundType getProbeBound(int position);
  bool CASBound(int position, BoundType old_value, BoundType new_value);
  VersionStateType getVersionState(int position, int index = 0);
  void setVersionState(int position, int index, VersionStateType version_state);
  bool CASVersionState(int position, int index, VersionStateType old_value, VersionStateType new_value);
  DataNode getData(int position, int index = 0);
  void setData(Key key, Value value, int position, int index = 0);

 private:
  int nprocs;
  int myrank;
  int table_length_each;
  HashFunctor hash_functor;

 private:
  MPI_Comm comm;
  MPI_Win bound_win;
  MPI_Win version_state_win;
  MPI_Win data_win;

 private:
  BoundType* bound_arr;
  VersionStateType* version_state_arr;
  DataNode* data_arr;

 private:
  class DataNode {
   public:
    DataNode() : key(), value() {};
    DataNode(Key key, Value value) : key(key), value(value) {};

   public:
    Key key;
    Value value;
  };
};

template <typename Key, typename Value, typename HashFunctor>
inline PurcellHarrisHashMap<Key, Value, HashFunctor>::PurcellHarrisHashMap(MPI_Comm comm, int table_length_each)
    : table_length_each(table_length_each), comm(comm) {
  MPI_Comm_size(comm, &this->nprocs);
  MPI_Comm_rank(comm, &this->myrank);

  this->bound_arr = new BoundType[this->table_length_each]();
  MPI_Win_create(this->bound_arr, (this->table_length_each * sizeof(BoundType)), sizeof(BoundType),
                 MPI_INFO_NULL, this->comm, &this->bound_win);

  this->version_state_arr = new VersionStateType[this->table_length_each]();
  MPI_Win_create(this->version_state_arr, (this->table_length_each * sizeof(VersionStateType)), sizeof(VersionStateType),
                 MPI_INFO_NULL, this->comm, &this->version_state_win);

  this->data_arr = new DataNode[this->table_length_each]();
  MPI_Win_create(this->data_arr, (this->table_length_each * sizeof(DataNode)), sizeof(DataNode),
                 MPI_INFO_NULL, this->comm, &this->data_win);

  MPI_Win_fence(0, this->bound_win);
  MPI_Win_fence(0, this->version_state_win);
  MPI_Win_fence(0, this->data_win);
  MPI_Win_lock_all(0, this->bound_win);
  MPI_Win_lock_all(0, this->version_state_win);
  MPI_Win_lock_all(0, this->data_win);
}
template <typename Key, typename Value, typename HashFunctor>
inline PurcellHarrisHashMap<Key, Value, HashFunctor>::~PurcellHarrisHashMap() {
  int flag;
  delete[] this->bound_arr;
  delete[] this->version_state_arr;
  delete[] this->data_arr;
  MPI_Finalized(&flag);
  if (!flag) {
    MPI_Win_unlock_all(this->bound_win);
    MPI_Win_unlock_all(this->version_state_win);
    MPI_Win_unlock_all(this->data_win);
    MPI_Win_free(&this->bound_win);
    MPI_Win_free(&this->version_state_win);
    MPI_Win_free(&this->data_win);
  }
}
template <typename Key, typename Value, typename HashFunctor>
inline bool PurcellHarrisHashMap<Key, Value, HashFunctor>::insert(Key key, Value value) {
#ifdef DEBUG
  std::cout << "[[Rank = " << this->myrank << "]]: " << "insert <" << key << "," << value << ">" << std::endl;
#endif  // DEBUG
  int position = this->hash_functor(key) % (this->nprocs * this->table_length_each);
  int i = -1;
  VersionStateType version_state;
  int version;
  bool result;
  do {
    // TODO: need to limit number of probing.
    if (++i > (this->table_length_each * this->nprocs)) {
      std::cout << "[[Rank = " << this->myrank << "]: table full, cant insert" << std::endl;
      MPI_Abort(this->comm, 1);
    }
    version_state = this->getVersionState(position, i);
    version = this->getVersion(version_state);
  } while (this->CASVersionState(position, i, this->makeVersionState(version, 0), this->makeVersionState(version, 1)) == false);
#ifdef DEBUG
  std::cout << "[[Rank = " << this->myrank << "]]: " << "insert <" << key << "," << value << ">, at <" << position << "," << i << ">" << std::endl;
#endif  // DEBUG
  this->setData(key, value, position, i);
  while (true) {
#ifdef DEBUG
    std::cout << "[[Rank = " << this->myrank << "]]: " << "flag1" << std::endl;
#endif  // DEBUG
    this->setVersionState(position, i, this->makeVersionState(version, 3));
    this->conditionallyRaiseBound(position, i);
    this->setVersionState(position, i, this->makeVersionState(version, 4));
    result = this->assist(key, position, i, version);
    if (this->getVersionState(position, i) != this->makeVersionState(version, 2))
      return true;
    if (result == false) {
      this->conditionallyLowerBound(position, i);
      this->setVersionState(position, i, this->makeVersionState(version + 1, 0));
      return false;
    }
    ++version;
  }
}
template <typename Key, typename Value, typename HashFunctor>
inline bool PurcellHarrisHashMap<Key, Value, HashFunctor>::get(Key key, Value& value) {
#ifdef DEBUG
  std::cout << "[[Rank = " << this->myrank << "]]: " << "get <" << key << ">" << std::endl;
#endif  // DEBUG
  int position = this->hash_functor(key) % (this->nprocs * this->table_length_each);
  int max_bound = this->getBound(position);
  VersionStateType version_state;
  int state;
  for (int i = 0; i < max_bound; ++i) {
    version_state = this->getVersionState(position, i);
    state = this->getState(version_state);
    if (this->isMember(state) == true) {
      DataNode data_node = this->getData(position, i);
      if (data_node.key == key) {
        value = data_node.value;
        if (version_state == this->getVersionState(position, i))
          return true;
      }
    }
  }
  return false;
}
template <typename Key, typename Value, typename HashFunctor>
inline bool PurcellHarrisHashMap<Key, Value, HashFunctor>::remove(Key key) {
#ifdef DEBUG
  std::cout << "[[Rank = " << this->myrank << "]]: " << "remove <" << key << ">" << std::endl;
#endif  // DEBUG
  int position = this->hash_functor(key) % (this->nprocs * this->table_length_each);
  int max_bound = this->getBound(position);
  VersionStateType version_state;
  int state;
  int version;
  for (int i = 0; i < max_bound; ++i) {
    version_state = this->getVersionState(position, i);
    state = this->getState(version_state);
    version = this->getVersion(version_state);
    if (this->isMember(state) == true) {
      DataNode data_node = this->getData(position, i);
      if (data_node.key == key) {
        if (this->CASVersionState(position, i, version_state, this->makeVersionState(version, 1)) == true) {
          this->conditionallyLowerBound(position, i);
          this->setVersionState(position, i, this->makeVersionState(this->getVersion(version_state) + 1, 0));
          return true;
        }
      }
    }
  }
  return false;
}
template <typename Key, typename Value, typename HashFunctor>
inline bool PurcellHarrisHashMap<Key, Value, HashFunctor>::assist(Key key, int position, int index, int version) {
#ifdef DEBUG
  std::cout << "[[Rank = " << this->myrank << "]]: " << "assist: " << key << ", at: <" << position << "," << index << "," << version << ">" << std::endl;
#endif  // DEBUG
  int max_bound = this->getBound(position);
  VersionStateType version_state_j;
  int state_j, version_j;
  for (int j = 0; j < max_bound; ++j) {
    if (index != j) {
      version_state_j = this->getVersionState(position, j);
      state_j = this->getState(version_state_j);
      version_j = this->getVersion(version_state_j);
      DataNode data_node_j = this->getData(position, j);
      if (this->isInserting(state_j) && data_node_j.key == key) {
        if (j < index) {
          if (this->getVersionState(position, j) == this->makeVersionState(version_j, 4)) {
            this->CASVersionState(position, index, this->makeVersionState(version, 4), this->makeVersionState(version, 2));
            return this->assist(key, position, j, version_j);
          }
        } else {
          if (this->getVersionState(position, index) == this->makeVersionState(version, 4)) {
            this->CASVersionState(position, j, this->makeVersionState(version_j, 4), this->makeVersionState(version_j, 2));
          }
        }
      }
      version_state_j = this->getVersionState(position, j);
      state_j = this->getState(version_state_j);
      version_j = this->getVersion(version_state_j);
      data_node_j = this->getData(position, j);
      if (this->isMember(state_j) && data_node_j.key == key) {
        if (this->getVersionState(position, j) == this->makeVersionState(version_j, 5)) {
          this->CASVersionState(position, index, this->makeVersionState(version, 4), this->makeVersionState(version, 2));
          return false;
        }
      }
    }
  }
  this->CASVersionState(position, index, this->makeVersionState(version, 4), this->makeVersionState(version, 5));
  return true;
}
template <typename Key, typename Value, typename HashFunctor>
inline void PurcellHarrisHashMap<Key, Value, HashFunctor>::conditionallyRaiseBound(int position, int index) {
  BoundType bound_old, bound_new;
#ifdef DEBUG2
  int debug_count = 0;
  int debug_max_count = 10;
#endif  // DEBUG2
  do {
#ifdef DEBUG2
    ++debug_count;
    if (debug_count >= debug_max_count) {
      if (debug_count == debug_max_count) {
        std::cout << "[Rank = " << this->myrank << "]: " << "raise bound fail loop 1." << std::endl;
      }
      MPI_Abort(this->comm, 1);
    }
#endif  // DEBUG2
    bound_old = this->getProbeBound(position);
    bound_new = this->makeProbeBound(std::max(this->getBoundFromProbeBound(bound_old), index), false);
  } while (this->CASBound(position, bound_old, bound_new) == false);
}
template <typename Key, typename Value, typename HashFunctor>
inline void PurcellHarrisHashMap<Key, Value, HashFunctor>::conditionallyLowerBound(int position, int index) {
  BoundType bound_old = this->getProbeBound(position);
  if (this->isScanning(bound_old) == true) {
    this->CASBound(position, bound_old, this->boundUnscan(bound_old));
  }
#ifdef DEBUG2
  int debug_count = 0;
  int debug_max_count = 10;
#endif  // DEBUG2
  if (index > 0) {
    while (this->CASBound(position, this->makeProbeBound(index, true), this->makeProbeBound(index, false)) == true) {
#ifdef DEBUG2
      ++debug_count;
      if (debug_count >= debug_max_count) {
        if (debug_count == debug_max_count) {
          std::cout << "[Rank = " << this->myrank << "]: " << "lower bound fail loop 1." << std::endl;
        }
        MPI_Abort(this->comm, 1);
      }
#endif  // DEBUG2
      int i = index - 1;
      while ((i > 0) && !this->doesBucketContainCollision(position, i)) {
        --i;
      }
      this->CASBound(position, this->makeProbeBound(index, true), this->makeProbeBound(i, false));
    }
  }
}
template <typename Key, typename Value, typename HashFunctor>
inline int PurcellHarrisHashMap<Key, Value, HashFunctor>::getBucket(int position, int index) {
  return ((position + index * (index + 1) / 2) % (this->nprocs * this->table_length_each));
}
template <typename Key, typename Value, typename HashFunctor>
inline bool PurcellHarrisHashMap<Key, Value, HashFunctor>::doesBucketContainCollision(int position, int index) {
  VersionStateType version_state1 = this->getVersionState(position, index);
  int state1 = this->getState(version_state1);
  if (this->isVisible(state1) || this->isInserting(state1) || this->isMember(state1)) {
    if ((int)(this->hash_functor(this->getData(position, index).key) % (this->nprocs * this->table_length_each)) == position) {
      VersionStateType version_state2 = this->getVersionState(position, index);
      int state2 = this->getState(version_state2);
      if (this->isVisible(state2) || this->isInserting(state2) || this->isMember(state2)) {
        if (this->getVersion(version_state1) == this->getVersion(version_state2))
          return true;
      }
    }
  }
  return false;
}
template <typename Key, typename Value, typename HashFunctor>
inline int PurcellHarrisHashMap<Key, Value, HashFunctor>::getBound(int position) {
  return this->getBoundFromProbeBound(this->getProbeBound(position));
}
template <typename Key, typename Value, typename HashFunctor>
inline int PurcellHarrisHashMap<Key, Value, HashFunctor>::getBoundFromProbeBound(BoundType bound) {
  return (bound >> 1);
}
template <typename Key, typename Value, typename HashFunctor>
inline bool PurcellHarrisHashMap<Key, Value, HashFunctor>::isScanning(BoundType bound) {
  return (bound & 1);
}
template <typename Key, typename Value, typename HashFunctor>
inline typename PurcellHarrisHashMap<Key, Value, HashFunctor>::BoundType
PurcellHarrisHashMap<Key, Value, HashFunctor>::boundUnscan(BoundType bound) {
  return (bound & (~1));
}
template <typename Key, typename Value, typename HashFunctor>
inline typename PurcellHarrisHashMap<Key, Value, HashFunctor>::BoundType
PurcellHarrisHashMap<Key, Value, HashFunctor>::makeProbeBound(int bound, bool scanning) {
  return ((bound << 1) | scanning);
}
template <typename Key, typename Value, typename HashFunctor>
inline int PurcellHarrisHashMap<Key, Value, HashFunctor>::getVersion(VersionStateType version_state) {
  return (version_state >> 3);
}
template <typename Key, typename Value, typename HashFunctor>
inline int PurcellHarrisHashMap<Key, Value, HashFunctor>::getState(VersionStateType version_state) {
  return (version_state & 7);
}
template <typename Key, typename Value, typename HashFunctor>
inline typename PurcellHarrisHashMap<Key, Value, HashFunctor>::VersionStateType
PurcellHarrisHashMap<Key, Value, HashFunctor>::makeVersionState(int version, int state) {
  return ((version << 3) | state);
}
template <typename Key, typename Value, typename HashFunctor>
inline bool PurcellHarrisHashMap<Key, Value, HashFunctor>::isVisible(int state) {
  return (state == 3);
}
template <typename Key, typename Value, typename HashFunctor>
inline bool PurcellHarrisHashMap<Key, Value, HashFunctor>::isInserting(int state) {
  return (state == 4);
}
template <typename Key, typename Value, typename HashFunctor>
inline bool PurcellHarrisHashMap<Key, Value, HashFunctor>::isMember(int state) {
  return (state == 5);
}
template <typename Key, typename Value, typename HashFunctor>
inline int PurcellHarrisHashMap<Key, Value, HashFunctor>::getRank(int position, int index) {
  return (this->getBucket(position, index) / (this->table_length_each));
}
template <typename Key, typename Value, typename HashFunctor>
inline int PurcellHarrisHashMap<Key, Value, HashFunctor>::getDisp(int position, int index) {
  return (this->getBucket(position, index) % (this->table_length_each));
}
template <typename Key, typename Value, typename HashFunctor>
inline typename PurcellHarrisHashMap<Key, Value, HashFunctor>::BoundType
PurcellHarrisHashMap<Key, Value, HashFunctor>::getProbeBound(int position) {
  BoundType result = 10;
  int rank = this->getRank(position);
  int disp = this->getDisp(position);
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(BoundType), MPI_CHAR, rank, disp, sizeof(BoundType), MPI_CHAR, MPI_NO_OP, this->bound_win);
  MPI_Win_flush(rank, this->bound_win);
  return result;
}
template <typename Key, typename Value, typename HashFunctor>
inline bool PurcellHarrisHashMap<Key, Value, HashFunctor>::CASBound(int position, BoundType old_value, BoundType new_value) {
  BoundType result;
  int rank = this->getRank(position);
  int disp = this->getDisp(position);
  MPI_Compare_and_swap(&new_value, &old_value, &result, MPI_UINT64_T, rank, disp, this->bound_win);
  MPI_Win_flush(rank, this->bound_win);
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]: " << "CAS Bound at: <" << position << "> : " << "Rank: <" << rank << "," << disp << ">: " << old_value << "->" << new_value << ": " << ((result == old_value) ? "succeed" : "fail") << std::endl;
#endif  // DEBUG
  return (result == old_value);
}
template <typename Key, typename Value, typename HashFunctor>
inline typename PurcellHarrisHashMap<Key, Value, HashFunctor>::VersionStateType
PurcellHarrisHashMap<Key, Value, HashFunctor>::getVersionState(int position, int index) {
  VersionStateType result = 10;
  int rank = this->getRank(position, index);
  int disp = this->getDisp(position, index);
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(VersionStateType), MPI_CHAR, rank, disp, sizeof(VersionStateType), MPI_CHAR, MPI_NO_OP, this->version_state_win);
  MPI_Win_flush(rank, this->version_state_win);
  return result;
}
template <typename Key, typename Value, typename HashFunctor>
inline void PurcellHarrisHashMap<Key, Value, HashFunctor>::setVersionState(int position, int index, VersionStateType version_state) {
  int rank = this->getRank(position, index);
  int disp = this->getDisp(position, index);
  MPI_Accumulate(&version_state, sizeof(VersionStateType), MPI_CHAR, rank, disp, sizeof(VersionStateType), MPI_CHAR, MPI_REPLACE, this->version_state_win);
  MPI_Win_flush(rank, this->version_state_win);
}
template <typename Key, typename Value, typename HashFunctor>
inline bool PurcellHarrisHashMap<Key, Value, HashFunctor>::CASVersionState(int position, int index, VersionStateType old_value, VersionStateType new_value) {
  VersionStateType result;
  int rank = this->getRank(position, index);
  int disp = this->getDisp(position, index);
  MPI_Compare_and_swap(&new_value, &old_value, &result, MPI_UINT64_T, rank, disp, this->version_state_win);
  MPI_Win_flush(rank, this->version_state_win);
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]: " << "CAS VS at: <" << position << "," << index << "> : " << "Rank: <" << rank << "," << disp << ">: " << old_value << "->" << new_value << ": " << ((result == old_value) ? "succeed" : "fail") << std::endl;
#endif  // DEBUG
  return (result == old_value);
}
template <typename Key, typename Value, typename HashFunctor>
inline typename PurcellHarrisHashMap<Key, Value, HashFunctor>::DataNode
PurcellHarrisHashMap<Key, Value, HashFunctor>::getData(int position, int index) {
  DataNode result;
  int rank = this->getRank(position, index);
  int disp = this->getDisp(position, index);
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(DataNode), MPI_CHAR, rank, disp, sizeof(DataNode), MPI_CHAR, MPI_NO_OP, this->data_win);
  MPI_Win_flush(rank, this->data_win);
  return result;
}
template <typename Key, typename Value, typename HashFunctor>
inline void PurcellHarrisHashMap<Key, Value, HashFunctor>::setData(Key key, Value value, int position, int index) {
  DataNode data_node(key, value);
  int rank = this->getRank(position, index);
  int disp = this->getDisp(position, index);
  MPI_Accumulate(&data_node, sizeof(DataNode), MPI_CHAR, rank, disp, sizeof(DataNode), MPI_CHAR, MPI_REPLACE, this->data_win);
  MPI_Win_flush(rank, this->data_win);
}
}  // namespace ngu

#endif  // DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_PURCELL_HARRIS_HASH_MAP_H