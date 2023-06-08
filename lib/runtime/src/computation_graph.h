#ifndef _FLEXFLOW_RUNTIME_SRC_COMPUTATION_GRAPH_H
#define _FLEXFLOW_RUNTIME_SRC_COMPUTATION_GRAPH_H

#include "tensor.h"
#include "utils/expected.h"

namespace FlexFlow {

template <typename T> using or_error_msg = expected<T, std::string>;

struct TensorSourceInfo {
  Layer layer;
  int idx;
};

static_assert(std::is_copy_constructible<ComputationGraph>::value, "");
static_assert(std::is_move_constructible<ComputationGraph>::value, "");
static_assert(std::is_copy_assignable<ComputationGraph>::value, "");
static_assert(std::is_copy_constructible<ComputationGraph>::value, "");

} // namespace FlexFlow

MAKE_TYPEDEF_HASHABLE(::FlexFlow::tensor_guid_t);

#endif
