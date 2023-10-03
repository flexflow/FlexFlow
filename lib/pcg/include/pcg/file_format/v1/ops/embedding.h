#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_EMBEDDING_ATTRS_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_FILE_FORMAT_V1_OPS_EMBEDDING_ATTRS_H

#include "op-attrs/ops/embedding.h"
#include "pcg/file_format/v1/datatype.h"
#include "utils/visitable.h"

namespace FlexFlow {

enum class V1AggregateOp {
  SUM,
  AVG,
};

NLOHMANN_JSON_SERIALIZE_ENUM(V1AggregateOp,
                             {{V1AggregateOp::SUM, "SUM"},
                              {V1AggregateOp::AVG, "AVG"}});

V1AggregateOp to_v1(AggregateOp const &op);

struct V1EmbeddingAttrs {
  req<int> num_entries, out_channels;
  req<V1AggregateOp> aggr;
  req<V1DataType> data_type;
};
FF_VISITABLE_STRUCT(
    V1EmbeddingAttrs, num_entries, out_channels, aggr, data_type);
CHECK_IS_JSONABLE(V1EmbeddingAttrs);

V1EmbeddingAttrs to_v1(EmbeddingAttrs const &attrs);

} // namespace FlexFlow

#endif
