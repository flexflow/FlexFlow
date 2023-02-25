#ifndef _FLEXFLOW_OP_META_OPS_UNARY_OP_H
#define _FLEXFLOW_OP_META_OPS_UNARY_OP_H

#include "op-meta/parallel_tensor_shape.h"
#include "op-meta/ops/op_params.h"

namespace FlexFlow {
namespace opmeta {

class UnaryInput : public OpParamsInterface {
  bool is_valid(std::vector<ParallelTensorShape> const &) const override final;
  std::vector<ParallelTensorShape> output_shapes(std::vector<ParallelTensorShape> const &) const override final;

  virtual bool is_valid(ParallelTensorShape const &input_shape) const;
  virtual std::vector<ParallelTensorShape> output_shape(ParallelTensorShape const &) const = 0;
};

class UnaryOutput : public OpParamsInterface {
  int num_outputs(std::vector<ParallelTensorShape> const &) const override final;
  std::vector<ParallelTensorShape> output_shapes(std::vector<ParallelTensorShape> const &) const override final;

  virtual ParallelTensorShape output_shape(std::vector<ParallelTensorShape> const &) const = 0;
};

class UnaryOpParams : public OpParamsInterface {
  bool is_valid(std::vector<ParallelTensorShape> const &) const override final;
  int num_outputs(std::vector<ParallelTensorShape> const &) const override final;
  std::vector<ParallelTensorShape> output_shapes(std::vector<ParallelTensorShape> const &) const override final;

  virtual bool is_valid(ParallelTensorShape const &input_shape) const;
  virtual ParallelTensorShape output_shape(ParallelTensorShape const &input_shape) const = 0;
};

}
}

#endif 
