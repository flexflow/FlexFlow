#ifndef _FLEXFLOW_EMBEDDING_ATTRS_H
#define _FLEXFLOW_EMBEDDING_ATTRS_H

#include "core.h"
#include "op-attrs/datatype.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "op-attrs/tensor_shape.h"
#include "utils/fmt.h"
#include "utils/visitable.h"

namespace FlexFlow {

enum class AggregateOp {
  SUM,
  AVG,
};

std::string format_as(AggregateOp const &);
CHECK_FMTABLE(AggregateOp);

struct EmbeddingAttrs {
  int num_entries;
  int out_channels;
  AggregateOp aggr;
  req<DataType> data_type;
};
FF_VISITABLE_STRUCT(EmbeddingAttrs, num_entries, out_channels, aggr, data_type);
FF_VISIT_FMTABLE(EmbeddingAttrs);

CHECK_VALID_OP_ATTR(EmbeddingAttrs);

TensorShape get_weights_shape(EmbeddingAttrs const &, TensorShape const &);

} // namespace FlexFlow

#endif
