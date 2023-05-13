#ifndef _FF_OP_META_BATCH_MATMUL_ATTRS_H
#define _FF_OP_META_BATCH_MATMUL_ATTRS_H

#include "op-attrs/parallel_tensor_shape.h"
#include "core.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct BatchMatmulAttrs : public use_visitable_cmp<BatchMatmulAttrs> {
public:
  BatchMatmulAttrs() = delete;
  BatchMatmulAttrs(int a_seq_length_dim, int b_seq_length_dim);
public:
  int a_seq_length_dim, b_seq_length_dim;
};

}

VISITABLE_STRUCT(::FlexFlow::BatchMatmulAttrs, a_seq_length_dim, b_seq_length_dim);
MAKE_VISIT_HASHABLE(::FlexFlow::BatchMatmulAttrs);

namespace FlexFlow {

static_assert(is_valid_opattr<BatchMatmulAttrs>::value, "BatchMatmulAttrs must be a valid opattr (see core.h)");

}

#endif 
