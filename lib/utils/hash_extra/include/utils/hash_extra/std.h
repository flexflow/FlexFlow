#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_HASH_STD_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_HASH_STD_H

#include <unordered_set>
#include <vector>
#include <unordered_map>
#include <type_traits>
#include <tuple>
#include "hash-combine.h"
#include "utils/algorithms/sorting.h"

namespace std {

template <class... TupleArgs>
struct hash<std::tuple<TupleArgs...>> {
private:
  //  this is a termination condition
  //  N == sizeof...(TupleTypes)
  //
  template <size_t Idx, typename... TupleTypes>
  inline typename std::enable_if<Idx == sizeof...(TupleTypes), void>::type
      hash_combine_tup(size_t &seed,
                       std::tuple<TupleTypes...> const &tup) const {}

  //  this is the computation function
  //  continues till condition N < sizeof...(TupleTypes) holds
  //
  template <size_t Idx, typename... TupleTypes>
      inline typename std::enable_if < Idx<sizeof...(TupleTypes), void>::type
      hash_combine_tup(size_t &seed,
                       std::tuple<TupleTypes...> const &tup) const {
    hash_combine(seed, std::get<Idx>(tup));

    //  on to next element
    hash_combine_tup<Idx + 1>(seed, tup);
  }

public:
  size_t operator()(std::tuple<TupleArgs...> const &tupleValue) const {
    size_t seed = 0;
    //  begin with the first iteration
    hash_combine_tup<0>(seed, tupleValue);
    return seed;
  }
};

template <typename L, typename R>
struct hash<std::pair<L, R>> {
  size_t operator()(std::pair<L, R> const &p) const {
    size_t seed = 283746;

    hash_combine(seed, p.first);
    hash_combine(seed, p.second);

    return seed;
  }
};

template <typename T>
struct hash<std::vector<T>> {
  size_t operator()(std::vector<T> const &vec) const {
    size_t seed = 0;
    iter_hash(seed, vec.cbegin(), vec.cend());
    return seed;
  }
};

template <typename T>
struct hash<std::unordered_set<T>> {
  size_t operator()(std::unordered_set<T> const &s) const {
    auto sorted = sorted_by(s, ::FlexFlow::compare_by<T>([](T const &t) {
                              return get_std_hash(t);
                            }));
    return get_std_hash(sorted);
  }
};

template <typename K, typename V>
struct hash<std::unordered_map<K, V>> {
  size_t operator()(std::unordered_map<K, V> const &m) const {
    return get_std_hash(items(m));
  }
};

}

#endif
