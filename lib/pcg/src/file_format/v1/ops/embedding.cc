#include "pcg/file_format/v1/ops/embedding.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1AggregateOp to_v1(AggregateOp const &op) {
  NOT_IMPLEMENTED();
}

V1EmbeddingAttrs to_v1(EmbeddingAttrs const &a) {
  return {to_v1(a.num_entries),
          to_v1(a.out_channels),
          to_v1(a.aggr),
          to_v1(a.data_type)};
}

} // namespace FlexFlow
