#ifndef _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H
#define _FLEXFLOW_OPATTRS_TENSOR_SHAPE_H

#include "utils/stack_vector.h"
#include "ffconst.h"

namespace FlexFlow {

struct TensorShape {
  TensorShape() = delete;
  TensorShape(std::vector<size_t> const &dims, DataType data_type);

  template <size_t MAXSIZE>
  TensorShape(stack_vector<size_t, MAXSIZE> const &dims, DataType data_type)
    : dims(dims.start(), dims.end()), data_type(data_type)
  { }

  int num_dims() const;
public:
  DataType data_type;
  stack_vector<size_t, MAX_TENSOR_DIM> dims;
};

}

#endif
