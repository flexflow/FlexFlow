#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_NORM_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_OPS_BATCH_NORM_H

#include "utils/visitable.h"
#include "core.h"
#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

struct BatchNormAttrs : public use_visitable_cmp<BatchNormAttrs> {
public:
  BatchNormAttrs() = delete;
  explicit BatchNormAttrs(bool relu);
public:
  bool relu;
};

ParallelTensorShape get_output_shape(BatchNormAttrs const &);

}

VISITABLE_STRUCT(::FlexFlow::BatchNormAttrs, relu);
MAKE_VISIT_HASHABLE(::FlexFlow::BatchNormAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<BatchNormAttrs>::value, "BatchNormAttrs must be a valid opattr (see core.h)");
}

#endif
