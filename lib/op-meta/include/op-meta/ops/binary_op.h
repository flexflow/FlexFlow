#ifndef _FLEXFLOW_OP_META_OPS_BINARY_OP_H
#define _FLEXFLOW_OP_META_OPS_BINARY_OP_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {
namespace opmeta {

class BinaryOpParams : public OpParamsInterface {
  bool is_valid(std::vector<ParallelTensorShape> const &) const override final;
  int num_outputs(std::vector<ParallelTensorShape> const &) const override final;
  std::vector<ParallelTensorShape> output_shapes(std::vector<ParallelTensorShape> const &input_shapes) const override final;

  virtual bool is_valid(ParallelTensorShape const &, ParallelTensorShape const &) const;
  virtual ParallelTensorShape output_shape(ParallelTensorShape const &, ParallelTensorShape const &) const = 0;
};

}
}

#endif 
