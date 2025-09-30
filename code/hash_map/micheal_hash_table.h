#ifndef DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_MICHEAL_HASHTABLE_H
#define DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_MICHEAL_HASHTABLE_H

#ifdef DEBUG
#include <unistd.h>
#endif  // DEBUG

#include <mpi.h>

#include <algorithm>
#include <bitset>
#include <cmath>
#include <functional>
#include <iostream>
#include <list>
#include <random>

namespace ngu {
uint64_t splitmix64_2(uint64_t x) {
  x += 0x9e3779b97f4a7c15ULL;
  x = (x ^ (x >> 30)) * 0xbf58476d1ce4e5b9ULL;
  x = (x ^ (x >> 27)) * 0x94d049bb133111ebULL;
  x = x ^ (x >> 31);
  return x;
}
template <typename Key>
class Compare {
 public:
  bool operator()(Key key1, Key key2) const {
    return key1 < key2;
  };
};

template <typename Key>
class Hash2 {
 public:
  uint64_t operator()(Key key) const {
    std::hash<Key> h;
    return splitmix64_2(h(key) + 12345);
  };
};

template <typename Key, typename Value, typename HashFunctor = Hash2<Key>, typename KeyCompare = Compare<Key>>
class MichealHashTable {
  /// @brief rank(15 bit) - disp(48 bit) - tag(1 bit)
  using GPtr = uint64_t;
  class DataNode;

 private:
  static const GPtr null_gptr = 0;

 public:
  MichealHashTable(MPI_Comm comm = MPI_COMM_WORLD, int table_length_each = 1000);
  MichealHashTable(const MichealHashTable&) = delete;
  MichealHashTable& operator=(const MichealHashTable&) = delete;
  ~MichealHashTable();
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

 private:
  bool insertToList(GPtr head, Key key, Value value);
  bool getFromList(GPtr head, Key key, Value& value);
  bool removeFromList(GPtr head, Key key);
  /// @brief Find node from key in list.
  /// @param head Head of list.
  /// @param key Input key.
  /// @param value Output value if key found.
  /// @param prev Ouput prev node of node has key if found.
  /// @param curr Ouput node has key if found.
  /// @param next Ouput next node of node has key if found.
  /// @return True if found, otherwise False.
  bool findInList(GPtr head, Key key, Value& value, GPtr& prev, GPtr& curr, GPtr& next);
  /// @brief Get data node.
  /// @param node Address.
  /// @return Data node.
  DataNode getData(GPtr node);
  /// @brief Set data node.
  /// @param node Address.
  /// @param next Next node.
  /// @param key New key.
  /// @param value New value.
  void setData(GPtr node, GPtr next, Key key, Value value);
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
  GPtr allocateNode(Key key, Value value, GPtr next = null_gptr);
  /// @brief Scan deleted node for reuse.
  void scan();

 private:
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
  int table_length_each;
  HashFunctor hash_functor;
  KeyCompare key_order;
  std::size_t max_dlist_size = 10;

 private:
  MPI_Comm comm;
  MPI_Win table;
  MPI_Win data;

 private:
  GPtr* array_mem;
  std::list<DataNode*> data_mem;
  MPI_Win hp_win;
  GPtr* hp_arr;
  std::list<GPtr> plist;
  std::list<GPtr> rlist;
  std::list<GPtr> dlist;
  GPtr* head_ptr;

  class DataNode {
    using GPtr = MichealHashTable::GPtr;

   public:
    DataNode() : next(), key(), value() {};
    DataNode(Key key, Value value, GPtr next = MichealHashTable::null_gptr) : next(next), key(key), value(value) {};

   public:
    GPtr next;
    Key key;
    Value value;
  };
};

template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline MichealHashTable<Key, Value, HashFunctor, KeyCompare>::MichealHashTable(MPI_Comm comm, int table_length_each)
    : table_length_each(table_length_each), comm(comm), data_mem(), plist(), rlist(), dlist() {
#ifdef DEBUG
  std::cout << "[[" << this->myrank << "]]" << ": constructing..." << std::endl;
#endif  // DEBUG
  MPI_Comm_size(comm, &this->nprocs);
  MPI_Comm_rank(comm, &this->myrank);

  MPI_Win_create_dynamic(MPI_INFO_NULL, comm, &this->data);
  this->array_mem = new GPtr[this->table_length_each]();
  MPI_Win_attach(this->data, this->array_mem, this->table_length_each * sizeof(GPtr));
  MPI_Aint arr_addr;
  MPI_Get_address(array_mem, &arr_addr);
  GPtr arr_head = this->makeGPtr(this->myrank, arr_addr, 0);

  this->head_ptr = new GPtr(arr_head);
  MPI_Win_create(this->head_ptr, sizeof(GPtr), sizeof(GPtr), MPI_INFO_NULL, comm, &this->table);

  this->hp_arr = new GPtr[3]();
  MPI_Win_create(this->hp_arr, sizeof(GPtr) * 3, sizeof(GPtr), MPI_INFO_NULL, comm, &this->hp_win);
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

  MPI_Win_fence(0, this->table);
  MPI_Win_fence(0, this->data);
  MPI_Win_fence(0, this->hp_win);
  MPI_Win_lock_all(0, this->table);
  MPI_Win_lock_all(0, this->data);
  MPI_Win_lock_all(0, this->hp_win);
#ifdef DEBUG
  std::cout << "[[" << this->myrank << "]]" << ": construct successfully" << std::endl;
#endif  // DEBUG
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline MichealHashTable<Key, Value, HashFunctor, KeyCompare>::~MichealHashTable() {
  int flag;
  delete[] this->head_ptr;
  delete[] this->array_mem;
  for (typename std::list<DataNode*>::iterator it = data_mem.begin(); it != data_mem.end(); ++it) {
    delete *it;
  }
  delete[] this->hp_arr;
  MPI_Finalized(&flag);
  if (!flag) {
    MPI_Win_unlock_all(this->table);
    MPI_Win_unlock_all(this->data);
    MPI_Win_unlock_all(this->hp_win);
    MPI_Win_free(&this->table);
    MPI_Win_free(&this->data);
    MPI_Win_free(&this->hp_win);
  }
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool MichealHashTable<Key, Value, HashFunctor, KeyCompare>::insert(Key key, Value value) {
  int position = hash_functor(key) % (this->table_length_each * this->nprocs);
#ifdef DEBUG
  std::cout << "[[Rank = " << this->myrank << "]]: " << "insert <" << key << "," << value << ">. at: " << position << std::endl;
#endif  // DEBUG
  GPtr head_node = this->getHeadNode(position);
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "head node: " << head_node << std::endl;
#endif  // DEBUG
#ifdef DEBUG3
  bool result = this->insertToList(head_node, key, value);
  std::cout << "[[Rank = " << this->myrank << "]]: " << "insert <" << key << "," << value << ">. at: " << position << "/" << head_node << ". " << ((result == true) ? "succeed" : "fail") << std::endl;
  return result;
#endif  // DEBUG3
  return this->insertToList(head_node, key, value);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool MichealHashTable<Key, Value, HashFunctor, KeyCompare>::get(Key key, Value& value) {
  int position = hash_functor(key) % (this->table_length_each * this->nprocs);
  GPtr head_node = this->getHeadNode(position);
  return this->getFromList(head_node, key, value);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool MichealHashTable<Key, Value, HashFunctor, KeyCompare>::remove(Key key) {
  int position = hash_functor(key) % (this->table_length_each * this->nprocs);
  GPtr head_node = this->getHeadNode(position);
  return this->removeFromList(head_node, key);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool MichealHashTable<Key, Value, HashFunctor, KeyCompare>::insertToList(GPtr head, Key key, Value value) {
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "insert node at head node: " << head << std::endl;
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
      while (true) {
        sleep(5);
      }
    }
#endif  // DEBUG
    if (findInList(head, key, value_temp, prev, curr, next) == true) {
      result = false;
      break;
    }
    GPtr new_node = this->allocateNode(key, value, unmark(curr));
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
inline bool MichealHashTable<Key, Value, HashFunctor, KeyCompare>::getFromList(GPtr head, Key key, Value& value) {
  GPtr prev = null_gptr, curr = null_gptr, next = null_gptr;
  bool result = this->findInList(head, key, value, prev, curr, next);
  this->setHPtr(null_gptr, 0);
  this->setHPtr(null_gptr, 1);
  this->setHPtr(null_gptr, 2);
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool MichealHashTable<Key, Value, HashFunctor, KeyCompare>::removeFromList(GPtr head, Key key) {
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
      while (true) {
        sleep(5);
      }
    }
#endif  // DEBUG
    if (this->findInList(head, key, value_temp, prev, curr, next) == false) {
      result = false;
      break;
    }
    if (CASNext(curr, unmark(next), mark(next)) == false)
      continue;
    if (CASNext(prev, unmark(curr), unmark(next)) == true)
      this->safeFreeNode(unmark(curr));
    else
      this->findInList(head, key, value_temp, prev, curr, next);
    result = true;
    break;
  }
  this->setHPtr(null_gptr, 0);
  this->setHPtr(null_gptr, 1);
  this->setHPtr(null_gptr, 2);
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool MichealHashTable<Key, Value, HashFunctor, KeyCompare>::findInList(GPtr head, Key key, Value& value, GPtr& prev, GPtr& curr, GPtr& next) {
  // TODO
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
      while (true) {
        sleep(5);
      }
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
        while (true) {
          sleep(5);
        }
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
      Key curr_key = data_node.key;
      value = data_node.value;
      if (unmark(curr) != this->getNextNode(prev)) {
        flag = true;
        break;
      }
      if (isMarked(next) == false) {
        if (this->key_order(curr_key, key) == false) {
#ifdef DEBUG
          std::cout << "[Rank = " << this->myrank << "]: " << "get node from list node return: " << (curr_key == key) << "||" << prev << " -> " << curr << std::endl;
#endif  // DEBUG
          return (curr_key == key);
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
inline typename MichealHashTable<Key, Value, HashFunctor, KeyCompare>::DataNode
MichealHashTable<Key, Value, HashFunctor, KeyCompare>::getData(GPtr node) {
  DataNode result;
  int rank = this->getRank(node);
  MPI_Aint disp = this->getDisp(node);
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(DataNode), MPI_CHAR, rank, disp, sizeof(DataNode), MPI_CHAR, MPI_NO_OP, this->data);
  MPI_Win_flush(rank, this->data);
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void MichealHashTable<Key, Value, HashFunctor, KeyCompare>::setData(GPtr node, GPtr next, Key key, Value value) {
  DataNode data_node(key, value, next);
  int rank = this->getRank(node);
  MPI_Aint disp = this->getDisp(node);
  MPI_Accumulate(&data_node, sizeof(DataNode), MPI_CHAR, rank, disp, sizeof(DataNode), MPI_CHAR, MPI_REPLACE, this->data);
  MPI_Win_flush(rank, this->data);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void MichealHashTable<Key, Value, HashFunctor, KeyCompare>::safeFreeNode(GPtr node_to_free) {
  this->dlist.push_back(node_to_free);
  if (this->dlist.size() >= this->max_dlist_size)
    this->scan();
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void MichealHashTable<Key, Value, HashFunctor, KeyCompare>::freeNode(GPtr node_to_free) {
  this->plist.push_back(node_to_free);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename MichealHashTable<Key, Value, HashFunctor, KeyCompare>::GPtr
MichealHashTable<Key, Value, HashFunctor, KeyCompare>::allocateNode(Key key, Value value, GPtr next) {
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "alloc... " << std::endl;
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
  this->setData(result, next, key, value);
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "alloc success " << std::endl;
#endif  // DEBUG
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void MichealHashTable<Key, Value, HashFunctor, KeyCompare>::scan() {
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
inline typename MichealHashTable<Key, Value, HashFunctor, KeyCompare>::GPtr
MichealHashTable<Key, Value, HashFunctor, KeyCompare>::getHPtr(int rank, int number) {
  GPtr result = 2;
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(GPtr), MPI_CHAR, rank, number, sizeof(GPtr), MPI_CHAR, MPI_NO_OP, this->hp_win);
  MPI_Win_flush(rank, this->hp_win);
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline void MichealHashTable<Key, Value, HashFunctor, KeyCompare>::setHPtr(GPtr node, int number) {
  MPI_Accumulate(&node, sizeof(GPtr), MPI_CHAR, this->myrank, number, sizeof(GPtr), MPI_CHAR, MPI_REPLACE, this->hp_win);
  MPI_Win_flush(this->myrank, this->hp_win);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename MichealHashTable<Key, Value, HashFunctor, KeyCompare>::GPtr
MichealHashTable<Key, Value, HashFunctor, KeyCompare>::makeGPtr(int rank, MPI_Aint disp, int tag) {
  return ((uint64_t)rank << 49) | (disp << 1) | tag;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename MichealHashTable<Key, Value, HashFunctor, KeyCompare>::GPtr
MichealHashTable<Key, Value, HashFunctor, KeyCompare>::getHeadNode(int position) {
  int rank = position / this->table_length_each;
  int disp = position % this->table_length_each;

  // int rank = position % this->nprocs;
  // int disp = position / this->nprocs;
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "get head: at rank = " << rank << ", disp = " << disp << std::endl;
#endif  // DEBUG

  GPtr other_table = 3;
  MPI_Get_accumulate(NULL, 0, MPI_INT, &other_table, sizeof(GPtr), MPI_CHAR, rank, 0, sizeof(GPtr), MPI_CHAR, MPI_NO_OP, this->table);
  MPI_Win_flush(rank, this->table);

  MPI_Aint real_disp = this->getDisp(other_table) + disp * sizeof(GPtr);
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "get head: succeed." << std::endl;
#endif  // DEBUG
  return this->makeGPtr(rank, real_disp, 0);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline int MichealHashTable<Key, Value, HashFunctor, KeyCompare>::getRank(GPtr node) {
  return (node >> 49);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline MPI_Aint MichealHashTable<Key, Value, HashFunctor, KeyCompare>::getDisp(GPtr node) {
  // return ((node << 15) >> 16);
  return (node >> 1) & ((1ULL << 48) - 1);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool MichealHashTable<Key, Value, HashFunctor, KeyCompare>::CASNext(GPtr curr, GPtr old_next, GPtr new_next) {
#ifdef DEBUG
  std::cout << "[" << this->myrank << "]: " << ": CAS at: " << curr << ", " << old_next << "->" << new_next << std::endl;
#endif  // DEBUG
  GPtr result;
  int rank = this->getRank(curr);
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
inline typename MichealHashTable<Key, Value, HashFunctor, KeyCompare>::GPtr
MichealHashTable<Key, Value, HashFunctor, KeyCompare>::mark(GPtr node) {
  return (node | 1);
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename MichealHashTable<Key, Value, HashFunctor, KeyCompare>::GPtr
MichealHashTable<Key, Value, HashFunctor, KeyCompare>::unmark(GPtr node) {
  return (node & (~1));
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline typename MichealHashTable<Key, Value, HashFunctor, KeyCompare>::GPtr
MichealHashTable<Key, Value, HashFunctor, KeyCompare>::getNextNode(GPtr node) {
  GPtr result;
  int rank = this->getRank(node);
  MPI_Aint disp = this->getDisp(node);
  MPI_Get_accumulate(NULL, 0, MPI_INT, &result, sizeof(GPtr), MPI_CHAR, rank, disp, sizeof(GPtr), MPI_CHAR, MPI_NO_OP, this->data);
  MPI_Win_flush(rank, this->data);
#ifdef DEBUG
  std::cout << "[Rank = " << this->myrank << "]: " << "get next node: " << node << " -> " << result << std::endl;
#endif  // DEBUG
  return result;
}
template <typename Key, typename Value, typename HashFunctor, typename KeyCompare>
inline bool MichealHashTable<Key, Value, HashFunctor, KeyCompare>::isMarked(GPtr node) {
  return (node & 1);
}
}  // namespace ngu

#endif  // DISTRIBUTED_DATA_STRUCTURE_HASH_MAP_MICHEAL_HASHTABLE_H