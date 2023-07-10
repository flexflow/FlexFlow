#ifndef _FLEXFLOW_OP_META_OPS_TRANSPOSE_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_TRANSPOSE_ATTRS_H

#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct TransposeAttrs : use_visitable_cmp<TransposeAttrs> {
public:
  TransposeAttrs() = delete;
  explicit TransposeAttrs(stack_vector<ff_dim_t, MAX_TENSOR_DIM> const &perm);

public:
  stack_vector<ff_dim_t, MAX_TENSOR_DIM> perm;
};

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::TransposeAttrs, perm);
MAKE_VISIT_HASHABLE(::FlexFlow::TransposeAttrs);

#endif
