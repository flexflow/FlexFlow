#ifndef _FLEXFLOW_INCLUDE_UTILS_VISITABLE_H
#define _FLEXFLOW_INCLUDE_UTILS_VISITABLE_H

#include "utils/exception.h"
#include "utils/hash-utils.h"
#include "utils/required_core.h"
#include "utils/sequence.h"
#include "utils/tuple.h"
#include "utils/type_traits.h"
#include "utils/visitable_core.h"

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

template <typename C, typename... Args>
struct construct_visitor {
  construct_visitor(C &c, std::tuple<Args const &...> args)
      : c(c), args(args) {}

  std::size_t idx = 0;
  std::tuple<Args const &...> args;
  C &c;

  template <typename T>
  void operator()(char const *, T C::*ptr_to_member) {
    c.*ptr_to_member = any_cast<T const &>(get(args, idx));
    this->idx++;
  };
};

template <typename T, typename... Args>
void visit_construct(T &t, Args &&...args) {
  static_assert(is_visitable<T>::value, "Type must be visitable");
  static_assert(std::is_same<std::tuple<Args...>, visit_as_tuple_t<T>>::value,
                "");

  std::tuple<Args...> tup(std::forward<Args>(args)...);
  construct_visitor<T> vis{t, tup};
  visit_struct::visit_pointers<T>(vis);
}

template <typename T, typename... Args>
void visit_construct_tuple(T &t, visit_as_tuple_t<T> const &tup) {
  static_assert(is_visitable<T>::value, "Type must be visitable");

  construct_visitor<T> vis{t, tup};
  visit_struct::visit_pointers<T>(vis);
}

template <typename T, typename... Args>
T make_visitable(Args &&...args) {
  T t(std::forward<Args>(args)...);
  return t;
}

template <typename T, typename Tup, int... S>
T visitable_from_tuple_impl(seq<S...>, Tup const &tup) {
  return T{std::get<S>(tup)...};
}

template <typename T, typename... Args>
T visitable_from_tuple(std::tuple<Args...> const &t) {
  return visitable_from_tuple_impl<T>(seq_enumerate_args_t<Args...>{}, t);
};

template <typename T>
struct GetFunctor {
  GetFunctor(T const &t) : t(t) {}

  T const &t;

  template <int IDX>
  auto operator()(std::integral_constant<int, IDX> const &) const
      -> remove_req_t<decltype(visit_struct::get<IDX>(t))> {
    return visit_struct::get<IDX>(t);
  }
};

template <typename T>
visit_as_tuple_t<T> as_tuple(T const &t) {
  GetFunctor<T> func(t);
  return seq_transform(func, seq_enumerate_t<visit_as_tuple_t<T>>{});
}

template <typename T>
struct use_visitable_constructor {
  template <typename... Args,
            typename = typename std::enable_if<std::is_same<
                std::tuple<Args...>,
                typename visit_struct::type_at<0, T>::type>::value>::type>
  use_visitable_constructor(Args &&...args) {
    visit_construct<T, Args...>(*this, std::forward<Args>(args)...);
  }
};

template <typename T, typename Enable = void>
struct is_list_initializable_from_tuple : std::false_type {};

template <typename T, typename... Args>
struct is_list_initializable_from_tuple<T, std::tuple<Args...>>
    : is_list_initializable<T, Args...> {};

template <typename T, typename Enable = void>
struct is_visit_list_initializable
    : conjunction<is_visitable<T>,
                  is_list_initializable_from_tuple<T, visit_as_tuple_t<T>>> {};

template <typename T, typename Enable = void>
struct is_only_visit_list_initializable
    : conjunction<is_visit_list_initializable<T>,
                  negation<is_list_initializable_from_tuple<
                      T,
                      tuple_head_t<-1, visit_as_tuple_t<T>>>>> {};

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

template <typename T, typename Enable = void>
struct is_well_behaved_visitable_type
    : conjunction<is_visitable<T>,
                  is_well_behaved_value_type<T>,
                  is_visit_list_initializable<T>,
                  biconditional<is_equal<field_count<T>,
                                         std::integral_constant<size_t, 0>>,
                                std::is_default_constructible<T>>> {};

struct fmt_visitor {
  std::ostringstream &oss;

  template <typename T>
  void operator()(char const *field_name, T const &field_value) {
    oss << " " << field_name << "=" << field_value;
  }
};

template <typename T>
std::string visit_format(T const &t) {
  static_assert(is_visitable<T>::value,
                "visit_format can only be applied to visitable types");
  static_assert(elements_satisfy<is_fmtable, T>::value,
                "Visitable fields must be fmtable");

  std::ostringstream oss;
  oss << "<" << ::visit_struct::get_name<T>();
  visit_struct::for_each(t, fmt_visitor{oss});
  oss << ">";

  return oss.str();
}

template <typename T>
auto operator==(T const &lhs, T const &rhs)
    -> enable_if_t<conjunction<is_visitable<T>,
                               elements_satisfy<is_equal_comparable, T>>::value,
                   bool> {
  return as_tuple(lhs) == as_tuple(rhs);
}

template <typename T, typename TT>
auto operator==(T const &lhs, TT const &rhs) -> enable_if_t<
    conjunction<is_visitable<T>, std::is_convertible<TT, T>>::value,
    bool> {
  return lhs == static_cast<T>(rhs);
}

template <typename T, typename TT>
auto operator==(T const &lhs, TT const &rhs)
    -> enable_if_t<conjunction<is_visitable<TT>,
                               negation<is_visitable<T>>,
                               std::is_convertible<T, TT>>::value,
                   bool> {
  return static_cast<TT>(lhs) == rhs;
}

/* template <typename T, typename TT> */
/* auto operator==(TT const &lhs, T const &rhs) -> enable_if_t< */
/*   conjunction<is_visitable<T>, elements_satisfy<is_equal_comparable, T>,
 * std::is_convertible<TT, T>, */
/*   negation<conjunction<is_visitable<T>, elements_satisfy<is_equal_comparable,
 * T>, std::is_convertible<TT, T>>> */
/*   >::value, */
/* bool> { */
/*   return as_tuple(lhs) == as_tuple(rhs); */
/* } */

template <typename T>
auto operator!=(T const &lhs, T const &rhs) -> enable_if_t<
    conjunction<is_visitable<T>, elements_satisfy<is_neq_comparable, T>>::value,
    bool> {
  return as_tuple(lhs) != as_tuple(rhs);
}

template <typename T>
auto operator<(T const &lhs, T const &rhs) -> enable_if_t<
    conjunction<is_visitable<T>, elements_satisfy<is_lt_comparable, T>>::value,
    bool> {
  return as_tuple(lhs) < as_tuple(rhs);
}

template <typename T>
struct visitable_formatter : public ::fmt::formatter<std::string> {
  template <typename FormatContext>
  auto format(T const &t, FormatContext &ctx) const -> decltype(ctx.out()) {
    std::string fmted = visit_format(t);
    return formatter<std::string>::format(fmted, ctx);
  }
};

} // namespace FlexFlow

#define CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION(TYPENAME)       \
  CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION_NO_EQ(TYPENAME);      \
  CHECK_WELL_BEHAVED_VALUE_TYPE(TYPENAME);

#define CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION_NO_EQ(TYPENAME) \
  static_assert(is_visitable<TYPENAME>::value,                                 \
                #TYPENAME " is not visitable (this should never "              \
                          "happen--contact the FF developers)");               \
  static_assert(sizeof(visit_as_tuple_raw_t<TYPENAME>) == sizeof(TYPENAME),    \
                #TYPENAME " should be fully visitable");                       \
  CHECK_WELL_BEHAVED_VALUE_TYPE_NO_EQ(TYPENAME);

#define FF_VISIT_FMTABLE(TYPENAME)                                             \
  static_assert(is_visitable<TYPENAME>::value,                                 \
                #TYPENAME " must be visitable to use FF_VISIT_FMTABLE");       \
  static_assert(elements_satisfy<is_streamable, TYPENAME>::value,              \
                #TYPENAME "'s elements must use be streamable");               \
  }                                                                            \
  namespace fmt {                                                              \
  template <>                                                                  \
  struct formatter<::FlexFlow::TYPENAME>                                       \
      : ::FlexFlow::visitable_formatter<::FlexFlow::TYPENAME> {};              \
  }                                                                            \
  namespace FlexFlow {                                                         \
  static_assert(is_fmtable<TYPENAME>::value,                                   \
                #TYPENAME                                                      \
                " failed sanity check on is_fmtable and FF_VISIT_FMTABLE");    \
  static_assert(is_streamable<TYPENAME>::value,                                \
                #TYPENAME                                                      \
                " failed sanity check on is_streamable and FF_VISIT_FMTABLE");

#define CHECK_WELL_BEHAVED_VISIT_TYPE(TYPENAME)                                \
  CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION(TYPENAME);            \
  static_assert(is_visit_list_initializable<TYPENAME>::value,                  \
                #TYPENAME                                                      \
                " should be list-initialializable by the visit field types");

#define CHECK_CONSTRUCTION_NONEMPTY(TYPENAME)                                  \
  static_assert(is_only_visit_list_initializable<TYPENAME>::value,             \
                #TYPENAME                                                      \
                " should not be list-initialializable from any sub-tuples "    \
                "(you probably need to insert req<...>s)");                    \
  static_assert(!std::is_default_constructible<TYPENAME>::value,               \
                #TYPENAME " should not be default-constructible (you "         \
                          "probably need to insert req<...>s)");               \
  static_assert(is_visit_list_initializable<TYPENAME>::value,                  \
                #TYPENAME                                                      \
                " should be list-initialializable by the visit field types");

#define CHECK_CONSTRUCTION_EMPTY(TYPENAME)                                     \
  static_assert(std::is_default_constructible<TYPENAME>::value,                \
                #TYPENAME " should be default-constructible as it is empty")

#define FF_VISITABLE_STRUCT_NO_EQ_EMPTY(TYPENAME)                              \
  }                                                                            \
  VISITABLE_STRUCT_EMPTY(::FlexFlow::TYPENAME);                                \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME);                                   \
  namespace FlexFlow {                                                         \
  CHECK_CONSTRUCTION_EMPTY(TYPENAME);

#define FF_VISITABLE_STRUCT_NO_EQ_NONEMPTY(TYPENAME, ...)                      \
  }                                                                            \
  VISITABLE_STRUCT(::FlexFlow::TYPENAME, __VA_ARGS__);                         \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME);                                   \
  namespace FlexFlow {                                                         \
  CHECK_CONSTRUCTION_NONEMPTY(TYPENAME);

#define FF_VISITABLE_STRUCT_EMPTY(TYPENAME)                                    \
  }                                                                            \
  VISITABLE_STRUCT_EMPTY(::FlexFlow::TYPENAME);                                \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME);                                   \
  namespace FlexFlow {                                                         \
  CHECK_WELL_BEHAVED_VISIT_TYPE(TYPENAME);                                     \
  CHECK_CONSTRUCTION_EMPTY(TYPENAME);

#define FF_VISITABLE_STRUCT_NONEMPTY(TYPENAME, ...)                            \
  }                                                                            \
  VISITABLE_STRUCT(::FlexFlow::TYPENAME, __VA_ARGS__);                         \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME);                                   \
  namespace FlexFlow {                                                         \
  CHECK_WELL_BEHAVED_VISIT_TYPE(TYPENAME);                                     \
  CHECK_CONSTRUCTION_NONEMPTY(TYPENAME);

#define FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION_EMPTY(TYPENAME)           \
  }                                                                            \
  VISITABLE_STRUCT_EMPTY(::FlexFlow::TYPENAME);                                \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME);                                   \
  namespace FlexFlow {                                                         \
  CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION(TYPENAME);

#define FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION_NONEMPTY(TYPENAME, ...)   \
  }                                                                            \
  VISITABLE_STRUCT(::FlexFlow::TYPENAME, __VA_ARGS__);                         \
  MAKE_VISIT_HASHABLE(::FlexFlow::TYPENAME);                                   \
  namespace FlexFlow {                                                         \
  CHECK_WELL_BEHAVED_VISIT_TYPE_NONSTANDARD_CONSTRUCTION(TYPENAME);

#define MAKE_VISIT_HASHABLE(TYPENAME)                                          \
  namespace std {                                                              \
  template <>                                                                  \
  struct hash<TYPENAME> : ::FlexFlow::use_visitable_hash<TYPENAME> {};         \
  }                                                                            \
  static_assert(true, "")

// see https://gustedt.wordpress.com/2010/06/03/default-arguments-for-c99/
// for an explanation of how this works
#define _GET_VISITABLE_CASE_FROM_NUM_ARGS(...)                                 \
  VISIT_STRUCT_EXPAND(VISIT_STRUCT_PP_ARG_N(__VA_ARGS__,                       \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            1,                                 \
                                            0,                                 \
                                            0))
#define _VISITABLE_STRUCT_CASE_0(MACRO_BASE_NAME, a) MACRO_BASE_NAME##_EMPTY(a)
#define _VISITABLE_STRUCT_CASE_1(MACRO_BASE_NAME, ...)                         \
  MACRO_BASE_NAME##_NONEMPTY(__VA_ARGS__)
#define __DISPATCH_VISITABLE_CASE(MACRO_BASE_NAME, N, ...)                     \
  _VISITABLE_STRUCT_CASE_##N(MACRO_BASE_NAME, __VA_ARGS__)
#define _DISPATCH_VISITABLE_CASE(MACRO_BASE_NAME, N, ...)                      \
  __DISPATCH_VISITABLE_CASE(MACRO_BASE_NAME, N, __VA_ARGS__)
#define FF_VISITABLE_STRUCT(...)                                               \
  _DISPATCH_VISITABLE_CASE(FF_VISITABLE_STRUCT,                                \
                           _GET_VISITABLE_CASE_FROM_NUM_ARGS(__VA_ARGS__),     \
                           __VA_ARGS__)

#define FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(...)                      \
  _DISPATCH_VISITABLE_CASE(FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION,       \
                           _GET_VISITABLE_CASE_FROM_NUM_ARGS(__VA_ARGS__),     \
                           __VA_ARGS__)

#define FF_VISITABLE_STRUCT_NO_EQ(...)                                         \
  _DISPATCH_VISITABLE_CASE(FF_VISITABLE_STRUCT_NO_EQ,                          \
                           _GET_VISITABLE_CASE_FROM_NUM_ARGS(__VA_ARGS__),     \
                           __VA_ARGS__)

#endif
