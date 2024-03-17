#ifndef _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_MAP_INSTANCES_MONOID_H
#define _FLEXFLOW_LIB_UTILS_STACK_CONTAINERS_INCLUDE_UTILS_STACK_CONTAINERS_STACK_MAP_INSTANCES_MONOID_H

#include "utils/stack_containers/stack_map/stack_map.h"
#include <cstddef>

namespace FlexFlow {

template <typename T>
struct keys_functor {};

template <typename K, typename V, std::size_t MAXSIZE>
struct keys_functor<stack_map<K, V, MAXSIZE>> {
  using A = K;

  template <typename T>
  using F = stack_map<T, V, MAXSIZE>;

  template <typename Func>
  static F<std::invoke_result_t<Func, A>> fmap(stack_map<A, V> const &m,
                                               Func const &f) {
    using B = std::invoke_result_t<Func, A>;
    stack_map<B, V, MAXSIZE> result;
    for (auto const &[k, v] : m) {
      result[f(k)] = v;
    }
    return result;
  }
};

template <typename T>
struct values_functor {};

template <typename K, typename V, std::size_t MAXSIZE>
struct values_functor<stack_map<K, V, MAXSIZE>> {
  using A = V;

  template <typename T>
  using F = std::unordered_map<K, T, MAXSIZE>;

  template <typename Func>
  static F<std::invoke_result_t<Func, A>>
      fmap(stack_map<K, A, MAXSIZE> const &m, Func const &f) {
    using B = std::invoke_result_t<Func, A>;
    stack_map<K, B, MAXSIZE> result;
    for (auto const &[k, v] : m) {
      result[k] = f(v);
    }
    return result;
  }
};

template <typename T>
struct kv_functor {};

template <typename K, typename V, std::size_t MAXSIZE>
struct kv_functor<stack_map<K, V, MAXSIZE>> {
  using A = std::pair<K, V>;

  template <typename T, typename Enable = void>
  struct _F {};

  template <typename T1, typename T2>
  struct _F<std::pair<T1, T2>> : type_identity<stack_map<T1, T2, MAXSIZE>> {};

  template <typename T>
  using F = typename _F<T>::type;

  template <typename Func,
            typename = std::enable_if_t<
                is_isomorphic_to_pair_v<std::invoke_result_t<Func, A>>>>
  static F<std::invoke_result_t<Func, A>> fmap(F<A> const &m, Func const &f) {
    using B = std::invoke_result_t<Func, A>;
    F<B> result;
    for (auto const &a : m) {
      auto const [k_b, v_b] = f(a);
      result[k_b] = f(v_b);
    }
    return result;
  }
};

} // namespace FlexFlow

#endif
