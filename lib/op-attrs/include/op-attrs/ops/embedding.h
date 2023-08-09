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

struct EmbeddingAttrs {
  req<int> num_entries, out_channels;
  req<AggregateOp> aggr;
  req<DataType> data_type;
};
FF_VISITABLE_STRUCT(EmbeddingAttrs, num_entries, out_channels, aggr, data_type);
CHECK_VALID_OP_ATTR(EmbeddingAttrs);

TensorShape get_weights_shape(EmbeddingAttrs const &, TensorShape const &);

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::AggregateOp> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::AggregateOp o, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using namespace FlexFlow;

    string_view name = "unknown";
    switch (o) {
      case AggregateOp::SUM:
        name = "Sum";
        break;
      case AggregateOp::AVG:
        name = "Avg";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt

#endif
