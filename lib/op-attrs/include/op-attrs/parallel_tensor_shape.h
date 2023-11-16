#ifndef _OP_META_PARALLEL_TENSOR_SHAPE_H
#define _OP_META_PARALLEL_TENSOR_SHAPE_H

#include "datatype.h"
#include "op-attrs/tensor_shape.h"
#include "parallel_tensor_dims.h"
#include "utils/bidict.h"
#include "utils/record_formatter.h"
#include "utils/stack_vector.h"
#include "utils/visitable.h"
#include <unordered_map>
#include <vector>

namespace FlexFlow {

/**
 * @brief Represent the shape of a ParallelTensor.
 */
struct ParallelTensorShape {
  ParallelTensorShape(TensorShape const &);

  template <typename Dims>
  ParallelTensorShape(Dims const &dims, DataType data_type)
      : dims(dims), data_type(data_type) {}

  int num_dims() const;

  ParallelDim const &at(ff_dim_t const &) const;
  ParallelDim &at(ff_dim_t const &);
  ParallelDim const &operator[](ff_dim_t const &) const;
  ParallelDim &operator[](ff_dim_t const &);

public:
  req<ParallelTensorDims> dims;
  req<DataType> data_type;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(ParallelTensorShape, dims, data_type);

TensorShape get_piece_shape(ParallelTensorShape const &);
int get_num_replica_dims(ParallelTensorShape const &);
int get_num_replicas(ParallelTensorShape const &);

bool is_valid(ParallelTensorShape const &);

TensorShape get_tensor_shape_unsafe(ParallelTensorShape const &);
std::vector<TensorShape>
    get_tensor_shapes_unsafe(std::vector<ParallelTensorShape> const &);

} // namespace FlexFlow

#endif
