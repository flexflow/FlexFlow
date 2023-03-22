#ifndef _FLEXFLOW_OP_META_OPS_OP_PARAMS_H
#define _FLEXFLOW_OP_META_OPS_OP_PARAMS_H

#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

struct OpAttrsInterface {
  virtual int num_outputs(std::vector<ParallelTensorShape> const &inputs) const;
  virtual std::vector<ParallelTensorShape> output_shapes(std::vector<ParallelTensorShape> const &inputs) const = 0;
  virtual bool is_valid(std::vector<ParallelTensorShape> const &inputs) const = 0;
  virtual OperatorType op_type() const = 0;
};

}

#endif
