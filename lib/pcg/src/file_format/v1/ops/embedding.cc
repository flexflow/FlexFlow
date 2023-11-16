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

AggregateOp from_v1(V1AggregateOp const &vop) {
  // There should be a better way of doing this.
  switch (vop) {
    case V1AggregateOp::SUM:
      return AggregateOp::SUM;
    case V1AggregateOp::AVG:
      return AggregateOp::AVG;
    default:
      NOT_REACHABLE();
  }
}

V1EmbeddingAttrs to_v1(EmbeddingAttrs const &a) {
  return {a.num_entries, a.out_channels, to_v1(a.aggr), to_v1(a.data_type)};
}

EmbeddingAttrs from_v1(V1EmbeddingAttrs const &va) {
  return {
      va.num_entries, va.out_channels, from_v1(va.aggr), from_v1(va.data_type)};
}

} // namespace FlexFlow
