#ifndef _FLEXFLOW_INCLUDE_UTILS_VISITABLE_FUNCS_H
#define _FLEXFLOW_INCLUDE_UTILS_VISITABLE_FUNCS_H

#include "utils/visitable.h"

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
  static_assert(visit_struct::traits::is_visitable<T>::value, "Type must be visitable");

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
  static_assert(visit_struct::traits::is_visitable<T>::value, "Type must be visitable");

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
  static_assert(visit_struct::traits::is_visitable<T>::value, "Type must be visitable");

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
  static_assert(visit_struct::traits::is_visitable<T>::value, "Type must be visitable");

  hash_visitor vis;
  visit_struct::for_each(t, vis);
  return vis.result;
}

}

#endif
