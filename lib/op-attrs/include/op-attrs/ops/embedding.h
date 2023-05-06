#ifndef _FLEXFLOW_EMBEDDING_ATTRS_H
#define _FLEXFLOW_EMBEDDING_ATTRS_H

#include "op-attrs/ffconst.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/visitable.h"
#include "op-attrs/tensor_shape.h"
#include "core.h"

namespace FlexFlow {

struct EmbeddingAttrs : use_visitable_cmp<EmbeddingAttrs> {
public:
  EmbeddingAttrs() = delete;
  EmbeddingAttrs(int num_entries, int out_channels, AggrMode aggr, DataType data_type);
public:
  int num_entries, out_channels;
  AggrMode aggr;
  DataType data_type;
};

TensorShape get_weights_shape(EmbeddingAttrs const &, TensorShape const &);

}

VISITABLE_STRUCT(::FlexFlow::EmbeddingAttrs, num_entries, out_channels, aggr, data_type);
MAKE_VISIT_HASHABLE(::FlexFlow::EmbeddingAttrs);

namespace FlexFlow {
static_assert(is_valid_opattr<EmbeddingAttrs>::value, "EmbeddingAttrs must be a valid opattr (see core.h)"); }

#endif 
