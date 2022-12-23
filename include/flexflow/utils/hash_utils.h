#ifndef _FLEXFLOW_HASH_UTILS_H
#define _FLEXFLOW_HASH_UTILS_H

#include <functional>
#include <tuple>
#include <type_traits>
#include <vector>

// tuple hashing pulled from
// https://www.variadic.xyz/2018/01/15/hashing-stdpair-and-stdtuple/
template <class T>
inline void hash_combine(std::size_t &seed, T const &v) {
  std::hash<T> hasher;
  seed ^= hasher(v) + 0x9e3779b9 + (seed << 6) + (seed >> 2);
}

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
    hash_combine(seed, vec.size());
    for (auto const &ele : vec) {
      hash_combine(seed, ele);
    }
    return seed;
  }
};
} // namespace std

#endif // _FLEXFLOW_HASH_UTILS_H
