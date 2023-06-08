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

struct EmbeddingAttrs : public use_visitable_cmp<EmbeddingAttrs> {
public:
  EmbeddingAttrs() = delete;
  EmbeddingAttrs(int num_entries,
                 int out_channels,
                 AggregateOp aggr,
                 DataType data_type);

public:
  int num_entries, out_channels;
  AggregateOp aggr;
  DataType data_type;
};

TensorShape get_weights_shape(EmbeddingAttrs const &, TensorShape const &);

} // namespace FlexFlow

VISITABLE_STRUCT(
    ::FlexFlow::EmbeddingAttrs, num_entries, out_channels, aggr, data_type);
MAKE_VISIT_HASHABLE(::FlexFlow::EmbeddingAttrs);

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

namespace FlexFlow {
static_assert(is_valid_opattr<EmbeddingAttrs>::value,
              "EmbeddingAttrs must be a valid opattr (see core.h)");
}

#endif
