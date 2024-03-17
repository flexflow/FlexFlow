#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_HASH_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_OPERATIONS_HASH_H

#include "utils/type_traits_extra/is_hashable.h"
#include "utils/type_traits_extra/metafunction/elements_satisfy.h"
#include "utils/visitable/type/traits/is_visitable.h"
#include "visit_struct/visit_struct.hpp"
#include <functional>

namespace FlexFlow {

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

template <typename T>
struct use_visitable_hash {
  std::size_t operator()(T const &t) const {
    return visit_hash(t);
  }
};

#define MAKE_VISIT_HASHABLE(TYPENAME)                                          \
  namespace std {                                                              \
  template <>                                                                  \
  struct hash<TYPENAME> : ::FlexFlow::use_visitable_hash<TYPENAME> {};         \
  }                                                                            \
  static_assert(true, "")

} // namespace FlexFlow

#endif
