#include "op-attrs/parallel_tensor_shape.h"
#include "utils/containers.h"
#include "utils/hash-utils.h"

namespace FlexFlow {

int num_dims(ParallelTensorShape const &s) {
  return num_dims(s.dims);
}

int get_num_replica_dims(ParallelTensorShape const &shape) {
  return get_num_replica_dims(shape.dims);
}

int get_num_replicas(ParallelTensorShape const &shape) {
  return product(
      transform(filter(as_vector(shape.dims), is_replica_dim),
                [](ParallelDim const &d) -> int { return d.degree; }));
}

bool is_valid(ParallelTensorShape const &shape) {
  return is_valid(shape.dims);
}

ParallelDim dim_at_idx(ParallelTensorShape const &s, ff_dim_t d) {
  return dim_at_idx(s.dims, d);
}

ParallelDim &dim_at_idx(ParallelTensorShape &s, ff_dim_t d) {
  return dim_at_idx(s.dims, d);
}

ParallelTensorShape lift_to_parallel(TensorShape const &s) {
  return {lift_to_parallel(s.dims), s.data_type};
}

TensorShape get_tensor_shape_unsafe(ParallelTensorShape const &) {
  NOT_IMPLEMENTED();
}
} // namespace FlexFlow
