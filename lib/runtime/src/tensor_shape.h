#ifndef _FLEXFLOW_RUNTIME_SRC_TENSOR_SHAPE_H
#define _FLEXFLOW_RUNTIME_SRC_TENSOR_SHAPE_H

#include <cstddef>
#include "utils/stack_vector.h"
#include "op-attrs/ffconst.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

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

}

#endif
