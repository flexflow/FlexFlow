#ifndef _FLEXFLOW_OP_META_OPS_OP_PARAMS_H
#define _FLEXFLOW_OP_META_OPS_OP_PARAMS_H

#include "op-meta/parallel_tensor_shape.h"

namespace FlexFlow {

struct OpParamsInterface {
  virtual int num_outputs(std::vector<ParallelTensorShape> const &inputs) const = 0;
  virtual bool is_valid(std::vector<ParallelTensorShape> const &inputs) const = 0;
  virtual OperatorType op_type() const = 0;
};

}

#endif
