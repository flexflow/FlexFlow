#ifndef _FLEXFLOW_RUNTIME_SRC_TENSOR_SHAPE_H
#define _FLEXFLOW_RUNTIME_SRC_TENSOR_SHAPE_H

#include "kernels/legion_dim.h"
#include "op-attrs/datatype.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/tensor_shape.dtg.h"
#include "utils/stack_vector.h"
#include <cstddef>
#include "utils/visitable.h"

namespace FlexFlow {

// TODO FIXME @lockshaw remove inheritance from legion tensor dims
struct LegionTensorShape : public use_visitable_cmp<LegionTensorShape>,
                           public LegionTensorDims {
  LegionTensorShape() = delete;
  LegionTensorShape(std::vector<size_t> const &dims, DataType data_type);
  LegionTensorShape(TensorShape const &);

  template <size_t MAXSIZE>
  LegionTensorShape(stack_vector<size_t, MAXSIZE> const &dims,
                    DataType data_type)
      : LegionTensorDims(dims.start(), dims.end()), data_type(data_type) {}

  operator TensorShape() const;

public:
  DataType data_type;
};

ff_dim_t to_ff(legion_dim_t, size_t num_dims);
legion_dim_t legion_dim_from_ff_dim(ff_dim_t, size_t num_dims);

ff_dim_t to_ff(legion_dim_t, TensorShape const &);
legion_dim_t legion_dim_from_ff_dim(ff_dim_t, TensorShape const &);

} // namespace FlexFlow

#endif
