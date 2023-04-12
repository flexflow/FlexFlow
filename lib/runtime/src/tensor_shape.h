#ifndef _FLEXFLOW_RUNTIME_SRC_TENSOR_SHAPE_H
#define _FLEXFLOW_RUNTIME_SRC_TENSOR_SHAPE_H

#include <cstddef>
#include "utils/stack_vector.h"
#include "op-attrs/ffconst.h"
#include "op-attrs/tensor_shape.h"
#include "op-attrs/ff_dim.h"

namespace FlexFlow {

struct legion_dim_t : strong_typedef<legion_dim_t, int> {
  using strong_typedef::strong_typedef;
};

struct LegionTensorShape {
  LegionTensorShape() = delete;
  LegionTensorShape(std::vector<size_t> const &dims, DataType data_type);
  LegionTensorShape(TensorShape const &);

  template <size_t MAXSIZE>
  LegionTensorShape(stack_vector<size_t, MAXSIZE> const &dims, DataType data_type)
    : dims(dims.start(), dims.end()), data_type(data_type)
  { }

  operator TensorShape() const;

  int num_dims() const;
  
public:
  DataType data_type;
  stack_vector<size_t, MAX_TENSOR_DIM> dims;
};

ff_dim_t to_ff(legion_dim_t, int num_dims);
legion_dim_t to_legion(ff_dim_t, int num_dims);

ff_dim_t to_ff(legion_dim_t, TensorShape const &);
legion_dim_t to_legion(ff_dim_t, TensorShape const &);

}

MAKE_TYPEDEF_HASHABLE(::FlexFlow::legion_dim_t);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::legion_dim_t, "legion_dim");

#endif
