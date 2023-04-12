#ifndef _FLEXFLOW_OP_META_OPS_TRANSPOSE_ATTRS_H
#define _FLEXFLOW_OP_META_OPS_TRANSPOSE_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

struct TransposeAttrs {
public:
  TransposeAttrs() = delete;
  explicit TransposeAttrs(stack_vector<ff_dim_t, MAX_TENSOR_DIM> const &perm);
public:
  stack_vector<ff_dim_t, MAX_TENSOR_DIM> perm;
};

}

VISITABLE_STRUCT(::FlexFlow::TransposeAttrs, perm);
MAKE_VISIT_HASHABLE(::FlexFlow::TransposeAttrs);

#endif 
