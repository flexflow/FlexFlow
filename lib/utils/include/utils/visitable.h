#ifndef _FLEXFLOW_INCLUDE_UTILS_VISITABLE_H
#define _FLEXFLOW_INCLUDE_UTILS_VISITABLE_H

#include "rapidcheck.h"
#include "utils/hash-utils.h"
#include "utils/type_traits.h"
#include "utils/visitable_core.h"
#include "utils/any.h"
#include "utils/tuple.h"
#include "utils/exception.h"
#include "utils/sequence.h"

namespace FlexFlow {

struct eq_visitor {
  bool result = true;

  template <typename T>
  void operator()(char const *, T const &t1, T const &t2) {
    result &= (t1 == t2);
  }
};

template <typename T>
bool visit_eq(T const &lhs, T const &rhs) {
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

template <typename T>
bool visit_neq(T const &lhs, T const &rhs) {
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

template <typename T>
bool visit_lt(T const &t1, T const &t2) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(elements_satisfy<is_lt_comparable, T>::value,
                "Values must be comparable via operator<");

  lt_visitor vis;
  visit_struct::for_each(t1, t2, vis);
  return vis.result;
}

struct hash_visitor {
  std::size_t result = 0;

  template <typename T>
  void operator()(char const *, T const &t1) {
    hash_combine(result, t1);
  }
};

template <typename T>
std::size_t visit_hash(T const &t) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(elements_satisfy<is_hashable, T>::value,
                "Values must be hashable");

  hash_visitor vis;
  visit_struct::for_each(t, vis);
  return vis.result;
}

template <typename C, typename ...Args>
struct construct_visitor {
  construct_visitor(C &c, std::tuple<Args const &...> args)
    : c(c), args(args) { }

  std::size_t idx = 0;
  std::tuple<Args const &...> args;
  C &c;

  template <typename T>
  void operator()(char const *, T C::* ptr_to_member) {
    c.*ptr_to_member = any_cast<T const &>(get(args, idx));
    this->idx++;
  };
};

template <typename T, typename ...Args>
void visit_construct(T &t, Args &&... args) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(std::is_same<std::tuple<Args...>, visit_as_tuple<T>>::value, "");

  std::tuple<Args...> tup(std::forward<Args>(args)...);
  construct_visitor<T> vis{t, tup};
  visit_struct::visit_pointers<T>(vis);
}

template <typename T, typename ...Args>
void visit_construct_tuple(T &t, visit_as_tuple<T> const &tup) {
  static_assert(is_visitable<T>::value, "Type must be visitable");

  construct_visitor<T> vis{t, tup};
  visit_struct::visit_pointers<T>(vis);
}

template <typename T, typename ...Args>
T make_visitable(Args && ...args) {
  T t(std::forward<Args>(args)...);
  return t;
}

template <typename T, typename ...Args>
T visitable_from_tuple(std::tuple<Args...> const &t) {
  using Idxs = typename seq_count<std::tuple_size<decltype(t)>::value>::type;

  return visitable_from_tuple<T>(Idxs{}, t);
};

template <typename T, typename Tup, int ...S>
T visitable_from_tuple_impl(seq<S...>, Tup const &tup) {
  return T{std::get<S>(tup)...};
}

template <typename T> 
struct use_visitable_constructor {
  template <typename ...Args, typename = typename std::enable_if<std::is_same<std::tuple<Args...>, typename visit_struct::type_at<0, T>::type>::value>::type>
  use_visitable_constructor(Args && ...args) {
    visit_construct<T, Args...>(*this, std::forward<Args>(args)...);
  }
};

template <typename T>
struct use_visitable_eq {
  friend bool operator==(T const &lhs, T const &rhs) {
    return visit_eq(lhs, rhs);
  }

  friend bool operator!=(T const &lhs, T const &rhs) {
    return visit_neq(lhs, rhs);
  }
};

template <typename T>
struct use_visitable_cmp : use_visitable_eq<T> {
  friend bool operator<(T const &lhs, T const &rhs) {
    return visit_lt(lhs, rhs);
  }
};

template <typename T>
struct use_visitable_hash {
  std::size_t operator()(T const &t) const {
    return visit_hash(t);
  }
};
} // namespace FlexFlow

namespace rc {

struct gen_visitor {
  template <typename Member>
  auto operator()(Member const &m) -> Gen<Member> {
    return gen::set(m);
  }
};

template <typename T>
Gen<T> build_visitable(T const &t) {
  static_assert(::FlexFlow::is_visitable<T>::value, "Type must be visitable");

  gen_visitor vis;
  return gen::build<T>(visit_struct::for_each(t, vis));
}

template <typename T>
struct Arbitrary<
    T,
    typename std::enable_if<::FlexFlow::is_visitable<T>::value>::type> {
  static Gen<T> arbitrary() {
    return build_visitable<T>();
  }
};

} // namespace rc

#define FF_VISITABLE_STRUCT_EMPTY(TYPENAME) \
  } \
  VISITABLE_STRUCT_EMPTY(::FlexFlow::TYPENAME); \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME); \
  namespace FlexFlow { \
  static_assert(true, "")

#define FF_VISITABLE_STRUCT_NONEMPTY(TYPENAME, ...) \
  } \
  VISITABLE_STRUCT(::FlexFlow::TYPENAME, __VA_ARGS__); \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME); \
  namespace FlexFlow { \
  static_assert(true, "")

#define MAKE_VISIT_HASHABLE(TYPENAME)                                          \
  namespace std {                                                              \
  template <>                                                                  \
  struct hash<TYPENAME> : ::FlexFlow::use_visitable_hash<TYPENAME> {};         \
  }                                                                            \
  static_assert(true, "")


// see https://gustedt.wordpress.com/2010/06/03/default-arguments-for-c99/
// for an explanation of how this works
#define _COND2(...) VISIT_STRUCT_EXPAND(VISIT_STRUCT_PP_ARG_N(__VA_ARGS__,  \
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  \
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  \
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  \
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  \
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  \
        2, 2, 2, 2, 2, 2, 2, 2, 2, 2,  \
        2, 2, 2, 2, 2, 2, 2, 2, 1, 0))
#define _ONE_OR_TWO_ARGS_1(a) FF_VISITABLE_STRUCT_EMPTY(a)
#define _ONE_OR_TWO_ARGS_2(...) FF_VISITABLE_STRUCT_NONEMPTY(__VA_ARGS__)
#define __ONE_OR_TWO_ARGS(N, ...) _ONE_OR_TWO_ARGS_ ## N (__VA_ARGS__)
#define _ONE_OR_TWO_ARGS(N, ...) __ONE_OR_TWO_ARGS(N, __VA_ARGS__)
#define ONE_OR_TWO_ARGS(...) _ONE_OR_TWO_ARGS(_COND2(__VA_ARGS__), __VA_ARGS__)
#define FF_VISITABLE_STRUCT(...) ONE_OR_TWO_ARGS(__VA_ARGS__)

#endif
