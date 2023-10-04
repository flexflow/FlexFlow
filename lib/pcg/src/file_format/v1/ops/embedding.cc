#include "pcg/file_format/v1/ops/embedding.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1AggregateOp to_v1(AggregateOp const &op) {
  // There should be a better way of doing this.
  switch (op) {
    case AggregateOp::SUM:
      return V1AggregateOp::SUM;
    case AggregateOp::AVG:
      return V1AggregateOp::AVG;
    default:
      NOT_REACHABLE();
  }
}

V1EmbeddingAttrs to_v1(EmbeddingAttrs const &a) {
  return {a.num_entries, a.out_channels, to_v1(a.aggr), to_v1(a.data_type)};
}

} // namespace FlexFlow
