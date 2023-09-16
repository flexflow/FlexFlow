#ifndef _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_UNORDERED_SET_H
#define _FLEXFLOW_LIB_UTILS_HASH_EXTRA_INCLUDE_UTILS_HASH_EXTRA_STD_UNORDERED_SET_H

namespace std {

template <typename T>
struct hash<std::unordered_set<T>> {
  size_t operator()(std::unordered_set<T> const &s) const {
    auto sorted = sorted_by(s, ::FlexFlow::compare_by<T>([](T const &t) {
                              return get_std_hash(t);
                            }));
    return get_std_hash(sorted);
  }
};


} // namespace FlexFlow

#endif
