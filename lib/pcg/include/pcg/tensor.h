#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_TENSOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_TENSOR_H

#include "create_grad.h"
#include "initializer.h"
#include "op-attrs/param_sync.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

struct Tensor {
  /* Tensor() = delete; */
  /* Tensor(TensorShape const &, */
  /*        CreateGrad create_gradients, */
  /*        optional<Initializer const &> initializer = nullopt, */
  /*        optional<ParamSync> sync_type = nullopt); */

  size_t get_volume() const;
  TensorShape get_shape() const;
  int num_dims() const;

  operator TensorShape() const;

public:
  TensorDims dims;
  DataType data_type;
  req<optional<Initializer>> initializer;
  req<bool> create_gradients;
  req<optional<ParamSync>> sync_type;
};
FF_VISITABLE_STRUCT(
    Tensor, dims, data_type, initializer, create_gradients, sync_type);

using Parameter = Tensor;

} // namespace FlexFlow

#endif
