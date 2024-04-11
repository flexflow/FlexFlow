#ifndef _OP_META_PARALLEL_TENSOR_SHAPE_H
#define _OP_META_PARALLEL_TENSOR_SHAPE_H

#include "op-attrs/tensor_shape.h"
#include <vector>
#include "op-attrs/parallel_tensor_shape_t.h"
#include "op-attrs/tensor_shape.h"

namespace FlexFlow {

int num_dims(ParallelTensorShape const &);
ParallelDim dim_at_idx(ParallelTensorShape const &, ff_dim_t);
ParallelDim &dim_at_idx(ParallelTensorShape &, ff_dim_t);

ParallelTensorShape lift_to_parallel(TensorShape const &);

TensorShape get_piece_shape(ParallelTensorShape const &);
int get_num_replica_dims(ParallelTensorShape const &);
int get_num_replicas(ParallelTensorShape const &);

bool is_valid(ParallelTensorShape const &);

TensorShape get_tensor_shape_unsafe(ParallelTensorShape const &);
std::vector<TensorShape>
    get_tensor_shapes_unsafe(std::vector<ParallelTensorShape> const &);

} // namespace FlexFlow

#endif
