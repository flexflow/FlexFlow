#include "doctest/doctest.h"
#include "op-attrs/operator_attrs.h"
#include "parallel_tensor_shape.h"
#include <rapidcheck.h>

using namespace FlexFlow;
using namespace rc;

ParallelTensorShape aggregate_parallel_helper(int n,
                                              FFOrdered<ParallelDim> const &gate_pred_assign_dims,
                                              FFOrdered<ParallelDim> const &full_gate_grad_dims,
                                              FFOrdered<ParallelDim> const &exp_dims) {
  AggregateAttrs attrs {n, lambda_bal: 1.0};
  ParallelTensorShape gate_preds {gate_pred_assign_dims, DataType::FLOAT};
  ParallelTensorShape gate_assign {gate_pred_assign_dims, DataType::INT32};
  ParallelTensorShape true_gate_assign {gate_pred_assign_dims, DataType::INT32}; 
  ParallelTensorShape full_gate_gradients {full_gate_grad_dims, DataType::FLOAT};
  std::vector<ParallelTensorShape> exp_preds;
  for (int i=0; i<n; ++i) {
    exp_preds.push_back({exp_dims, DataType::FLOAT});
  }
  assert(is_valid(attrs, gate_preds, gate_assign, true_gate_assign, full_gate_gradients, exp_preds));
  

  return get_output_shape(attrs, gate_preds, gate_assign, true_gate_assign, full_gate_gradients, exp_preds);
}

TEST_CASE("1,AggregateAttrs:get_output_shape") {
  int n = 12;
  int k = 4;
  int batch_size = 64;
  int rows = 16;
  int output_dim = 32;
  AggregateAttrs attrs {n, lambda_bal: 1.0};

  ParallelDim replica_dim {1, 1, true};
  FFOrdered<ParallelDim> gate_pred_assign_dims {{k, 1, false}, {batch_size, 1, false}, replica_dim};
  FFOrdered<ParallelDim> full_gate_grad_dims {{n, 1, false}, {batch_size, 1, false}, {1, 1, true}};
  FFOrdered<ParallelDim> exp_dims {{output_dim, 1, false}, {rows, 1, false}, {1, 1, true}};
  FFOrdered<ParallelDim> output_dims {{output_dim, 1, false}, {rows, 1, false}, replica_dim};
  ParallelTensorShape correct_output {output_dims, DataType::FLOAT};

  CHECK(aggregate_parallel_helper(n, gate_pred_assign_dims, full_gate_grad_dims, exp_dims) == correct_output);
}