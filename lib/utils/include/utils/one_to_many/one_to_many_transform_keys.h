#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_TRANSFORM_KEYS_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_ONE_TO_MANY_ONE_TO_MANY_TRANSFORM_KEYS_H

#include "utils/containers/transform.h"
#include "utils/one_to_many/one_to_many.h"

namespace FlexFlow {

template <typename L1,
          typename R,
          typename F,
          typename L2 = std::invoke_result_t<F, L1>>
OneToMany<L2, R> one_to_many_transform_keys(OneToMany<L1, R> const &input,
                                            F &&f) {
  std::set<std::pair<L1, R>> input_pairs =
      unstructured_relation_from_one_to_many(input);

  std::set<std::pair<L2, R>> result_pairs = transform(
      input_pairs, [&](std::pair<L1, R> const &p) -> std::pair<L2, R> {
        return {f(p.first), p.second};
      });

  OneToMany<L2, R> result =
      one_to_many_from_unstructured_relation(result_pairs);

  return result;
}

} // namespace FlexFlow

#endif
