#ifndef _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H
#define _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H

#include "datatype.h"
#include "op-attrs/dim_ordered.h"
#include "op-attrs/ff_dim.h"
#include "utils/stack_vector.h"
#include "utils/visitable.h"

namespace FlexFlow {

using TensorDims = FFOrdered<size_t>;

struct TensorShape {
  size_t at(ff_dim_t) const;
  size_t operator[](ff_dim_t) const;

public:
  req<TensorDims> dims;
  req<DataType> data_type;
};

FF_VISITABLE_STRUCT(TensorShape, dims, data_type);
FF_VISIT_FMTABLE(TensorShape);
CHECK_FMTABLE(TensorShape);

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::TensorDims> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::TensorDims dims, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using namespace FlexFlow;

    std::vector<size_t> v(dims.begin(), dims.end());
    return formatter<string_view>::format(fmt::to_string(v), ctx);
  }
};

} // namespace fmt

#endif
