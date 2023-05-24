#ifndef _FLEXFLOW_RUNTIME_SRC_TENSOR_SHAPE_H
#define _FLEXFLOW_RUNTIME_SRC_TENSOR_SHAPE_H

#include <cstddef>
#include "utils/stack_vector.h"
#include "op-attrs/datatype.h"
#include "op-attrs/tensor_shape.h"
#include "op-attrs/ff_dim.h"
#include "kernels/legion_dim.h"

namespace FlexFlow {

// TODO FIXME @lockshaw remove inheritance from legion tensor dims
struct LegionTensorShape : public use_visitable_cmp<LegionTensorShape>, 
                           public LegionTensorDims {
  LegionTensorShape() = delete;
  LegionTensorShape(std::vector<size_t> const &dims, DataType data_type);
  LegionTensorShape(TensorShape const &);

  template <size_t MAXSIZE>
  LegionTensorShape(stack_vector<size_t, MAXSIZE> const &dims, DataType data_type)
    : LegionTensorDims(dims.start(), dims.end()), data_type(data_type)
  { }

  operator TensorShape() const;
public:
  DataType data_type;
};

ff_dim_t to_ff(legion_dim_t, int num_dims);
legion_dim_t to_legion(ff_dim_t, int num_dims);

ff_dim_t to_ff(legion_dim_t, TensorShape const &);
legion_dim_t to_legion(ff_dim_t, TensorShape const &);

}

MAKE_TYPEDEF_HASHABLE(::FlexFlow::legion_dim_t);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::legion_dim_t, "legion_dim");

#endif
