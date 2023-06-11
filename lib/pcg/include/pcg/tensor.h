#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_TENSOR_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_TENSOR_H

#include "create_grad.h"
#include "initializer.h"
#include "op-attrs/param_sync.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

struct Tensor : public use_visitable_cmp<Tensor> {
  Tensor() = delete;
  Tensor(TensorShape const &,
         CreateGrad create_gradients,
         optional<Initializer const &> initializer = nullopt,
         optional<ParamSync> sync_type = nullopt);

  size_t get_volume() const;
  TensorShape get_shape() const;
  int num_dims() const;

  operator TensorShape() const;

public:
  TensorDims dims;
  DataType data_type;
  optional<Initializer> initializer;
  bool create_gradients;
  optional<ParamSync> sync_type;
};

using Parameter = Tensor;

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::Tensor,
                 dims,
                 data_type,
                 initializer,
                 create_gradients,
                 sync_type);
MAKE_VISIT_HASHABLE(::FlexFlow::Tensor);

namespace FlexFlow {
static_assert(is_well_behaved_value_type<Tensor>::value, "");
}

#endif
