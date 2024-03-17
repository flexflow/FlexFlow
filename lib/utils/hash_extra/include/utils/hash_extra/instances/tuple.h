#ifndef _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_TUPLE_H
#define _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_TUPLE_H

#include "utils/hash_extra/hash_combine.h"
#include <functional>
#include <tuple>

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
    using ::FlexFlow::hash_combine;

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

} // namespace std

#endif
