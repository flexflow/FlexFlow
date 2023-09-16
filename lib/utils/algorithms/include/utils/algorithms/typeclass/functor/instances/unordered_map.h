#ifndef _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_FUNCTOR_INSTANCES_UNORDERED_MAP_H
#define _FLEXFLOW_LIB_UTILS_ALGORITHMS_INCLUDE_UTILS_ALGORITHMS_TYPE_FUNCTOR_INSTANCES_UNORDERED_MAP_H

#include <unordered_map>
#include "utils/algorithms/typeclass/functor/functor.h"
#include "utils/type_traits_extra/type_list/type_list.h"
#include "utils/type_traits_extra/type_list/indexing.h"
#include "utils/type_traits_extra/type_list/concat.h"
#include "utils/type_traits_extra/type_list/apply.h"
#include "utils/type_traits_extra/is_isomorphic_to_pair.h"

namespace FlexFlow {

template <typename T> struct keys_functor { };

template <typename K, typename V> 
struct keys_functor<std::unordered_map<K, V>> { 
  using A = K;

  template <typename T>
  using F = std::unordered_map<T, V>;

  template <typename Func>
  static F<std::invoke_result_t<Func, A>> fmap(std::unordered_map<A, V> const &m, Func const &f) {
    using B = std::invoke_result_t<Func, A>;
    std::unordered_map<B, V> result;
    for (auto const &[k, v] : m) {
      result[f(k)] = v;
    }
    return result;
  }
};

template <typename T> struct values_functor { };

template <typename K, typename V> 
struct values_functor<std::unordered_map<K, V>> { 
  using A = V;

  template <typename T>
  using F = std::unordered_map<K, T>;

  template <typename Func>
  static F<std::invoke_result_t<Func, A>> fmap(std::unordered_map<K, A> const &m, Func const &f) {
    using B = std::invoke_result_t<Func, A>;
    std::unordered_map<K, B> result;
    for (auto const &[k, v] : m) {
      result[k] = f(v);
    }
    return result;
  }
};

template <typename T> struct kv_functor { };

template <typename K, typename V>
struct kv_functor<std::unordered_map<K, V>> {
  using A = std::pair<K, V>;

  template <typename T, typename Enable = void> struct _F { };

  template <typename T1, typename T2>
  struct _F<std::pair<T1, T2>> : type_identity<std::unordered_map<T1, T2>> {};
    
  template <typename T>
  using F = typename _F<T>::type;

  template <typename Func, typename = std::enable_if_t<is_isomorphic_to_pair_v<std::invoke_result_t<Func, A>>>>
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
