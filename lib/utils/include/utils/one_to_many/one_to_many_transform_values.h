#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_TRANSFORM_VALUES_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_TRANSFORM_VALUES_H

#include "utils/containers/transform.h"
#include "utils/one_to_many/one_to_many.h"

namespace FlexFlow {

template <typename L,
          typename R1,
          typename F,
          typename R2 = std::invoke_result_t<F, R1>>
OneToMany<L, R2> one_to_many_transform_values(OneToMany<L, R1> const &input,
                                              F f) {
  std::set<std::pair<L, R1>> input_pairs =
      unstructured_relation_from_one_to_many(input);

  std::set<std::pair<L, R2>> result_pairs = transform(
      input_pairs, [&](std::pair<L, R1> const &p) -> std::pair<L, R2> {
        return {p.first, f(p.second)};
      });

  return one_to_many_from_unstructured_relation(result_pairs);
}

} // namespace FlexFlow

#endif
