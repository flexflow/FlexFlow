#ifndef _FLEXFLOW_OP_META_VISIT_STRUCT_H
#define _FLEXFLOW_OP_META_VISIT_STRUCT_H

#include "visit_struct/visit_struct.hpp"
#include "hash-utils.h"

namespace FlexFlow {

struct eq_visitor {
  bool result = true;

  template <typename T>
  void operator()(const char *, T const &t1, T const &t2) {
    result &= (t1 == t2);
  }
};

template <typename T>
bool visit_eq(T const &lhs, T const &rhs) {
  eq_visitor vis;
  visit_struct::for_each(lhs, rhs, vis);
  return vis.result;
}

struct neq_visitor {
  bool result = false;

  template <typename T>
  void operator()(const char *, T const &t1, T const &t2) {
    result |= (t1 != t2);
  }
};

template <typename T>
bool visit_neq(T const &lhs, T const &rhs) {
  neq_visitor vis;
  visit_struct::for_each(lhs, rhs, vis);
  return vis.result;
}

struct lt_visitor {
  bool result = true;

  template <typename T>
  void operator()(const char *, const T & t1, const T & t2) {
    result = result && (t1 < t2);
  }
};

template <typename T>
bool visit_lt(const T & t1, const T & t2) {
  eq_visitor vis;
  visit_struct::for_each(t1, t2, vis);
  return vis.result;
}

struct hash_visitor {
  std::size_t result = 0;

  template <typename T>
  void operator()(const char *, T const &t1) {
    hash_combine(result, t1);
  }
};

template <typename T>
std::size_t visit_hash(T const &t) {
  hash_visitor vis;
  visit_struct::for_each(t, vis);
  return vis.result;
}

}

#endif 
