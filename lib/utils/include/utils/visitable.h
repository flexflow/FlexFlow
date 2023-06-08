#ifndef _FLEXFLOW_INCLUDE_UTILS_VISITABLE_H
#define _FLEXFLOW_INCLUDE_UTILS_VISITABLE_H

#include "rapidcheck.h"
#include "utils/hash-utils.h"
#include "utils/type_traits.h"

namespace FlexFlow {

struct eq_visitor {
  bool result = true;

  template <typename T>
  void operator()(char const *, T const &t1, T const &t2) {
    result &= (t1 == t2);
  }
};

template <typename T> bool visit_eq(T const &lhs, T const &rhs) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(elements_satisfy<is_equal_comparable, T>::value,
                "Values must be comparable via operator==");

  eq_visitor vis;
  visit_struct::for_each(lhs, rhs, vis);
  return vis.result;
}

struct neq_visitor {
  bool result = false;

  template <typename T>
  void operator()(char const *, T const &t1, T const &t2) {
    result |= (t1 != t2);
  }
};

template <typename T> bool visit_neq(T const &lhs, T const &rhs) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(elements_satisfy<is_neq_comparable, T>::value,
                "Values must be comparable via operator!=");

  neq_visitor vis;
  visit_struct::for_each(lhs, rhs, vis);
  return vis.result;
}

struct lt_visitor {
  bool result = true;

  template <typename T>
  void operator()(char const *, T const &t1, T const &t2) {
    result = result && (t1 < t2);
  }
};

template <typename T> bool visit_lt(T const &t1, T const &t2) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(elements_satisfy<is_lt_comparable, T>::value,
                "Values must be comparable via operator<");

  lt_visitor vis;
  visit_struct::for_each(t1, t2, vis);
  return vis.result;
}

struct hash_visitor {
  std::size_t result = 0;

  template <typename T> void operator()(char const *, T const &t1) {
    hash_combine(result, t1);
  }
};

template <typename T> std::size_t visit_hash(T const &t) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(elements_satisfy<is_hashable, T>::value,
                "Values must be hashable");

  hash_visitor vis;
  visit_struct::for_each(t, vis);
  return vis.result;
}

template <typename T> struct use_visitable_eq {
  friend bool operator==(T const &lhs, T const &rhs) {
    return visit_eq(lhs, rhs);
  }

  friend bool operator!=(T const &lhs, T const &rhs) {
    return visit_neq(lhs, rhs);
  }
};

template <typename T> struct use_visitable_cmp : use_visitable_eq<T> {
  friend bool operator<(T const &lhs, T const &rhs) {
    return visit_lt(lhs, rhs);
  }
};

template <typename T> struct use_visitable_hash {
  std::size_t operator()(T const &t) const { return visit_hash(t); }
};
} // namespace FlexFlow

namespace rc {

struct gen_visitor {
  template <typename Member> auto operator()(Member const &m) -> Gen<Member> {
    return gen::set(m);
  }
};

template <typename T> Gen<T> build_visitable(T const &t) {
  static_assert(::FlexFlow::is_visitable<T>::value, "Type must be visitable");

  gen_visitor vis;
  return gen::build<T>(visit_struct::for_each(t, vis));
}

template <typename T>
struct Arbitrary<
    T, typename std::enable_if<::FlexFlow::is_visitable<T>::value>::type> {
  static Gen<T> arbitrary() { return build_visitable<T>(); }
};

} // namespace rc

#define MAKE_VISIT_HASHABLE(TYPENAME)                                          \
  namespace std {                                                              \
  template <>                                                                  \
  struct hash<TYPENAME> : ::FlexFlow::use_visitable_hash<TYPENAME> {};         \
  }                                                                            \
  static_assert(true, "")

#endif
