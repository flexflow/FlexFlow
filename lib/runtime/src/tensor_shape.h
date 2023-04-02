#ifndef _FLEXFLOW_RUNTIME_SRC_TENSOR_SHAPE_H
#define _FLEXFLOW_RUNTIME_SRC_TENSOR_SHAPE_H

#include <cstddef>
#include "utils/stack_vector.h"
#include "op-attrs/ffconst.h"

namespace FlexFlow {

struct TensorShape;
struct LegionTensorShape;

struct TensorShape {
  TensorShape() = delete;
  TensorShape(std::vector<size_t> const &dims, DataType data_type);
  TensorShape(LegionTensorShape const &);

  template <size_t MAXSIZE>
  TensorShape(stack_vector<size_t, MAXSIZE> const &dims, DataType data_type)
    : dims(dims.start(), dims.end()), data_type(data_type)
  { }

  int num_dims() const;

public:
  DataType data_type;
  stack_vector<size_t, MAX_TENSOR_DIM> dims;
};

struct LegionTensorShape {
  LegionTensorShape() = delete;
  LegionTensorShape(std::vector<size_t> const &dims, DataType data_type);
  LegionTensorShape(TensorShape const &);

  template <size_t MAXSIZE>
  LegionTensorShape(stack_vector<size_t, MAXSIZE> const &dims, DataType data_type)
    : dims(dims.start(), dims.end()), data_type(data_type)
  { }

  int num_dims() const;
  
public:
  DataType data_type;
  stack_vector<size_t, MAX_TENSOR_DIM> dims;
};

}

#endif
