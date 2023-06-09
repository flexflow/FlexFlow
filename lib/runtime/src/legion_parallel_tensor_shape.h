#ifndef _FLEXFLOW_RUNTIME_SRC_LEGION_PARALLEL_TENSOR_SHAPE_H
#define _FLEXFLOW_RUNTIME_SRC_LEGION_PARALLEL_TENSOR_SHAPE_H

#include "op-attrs/parallel_tensor_shape.h"

namespace FlexFlow {

struct LegionParallelTensorShape {
  LegionParallelTensorShape() = delete;
  LegionParallelTensorShape(ParallelTensorShape const &);

  template <typename C>
  LegionParallelTensorShape(C const &dims, DataType data_type)
      : dims(dims), data_type(data_type) {}

  bool operator==(LegionParallelTensorShape const &) const;
  bool operator!=(LegionParallelTensorShape const &) const;

  operator ParallelTensorShape() const;

  /* RecordFormatter as_dot() const; */

  /* size_t get_piece_size() const; */
  bool is_valid() const;

  /* int get_num_replica_dims() const; */
  /* int get_num_replicas() const; */

  ParallelDim const &at(int index) const;
  ParallelDim &at(int index);

  size_t size() const;
  size_t num_dims() const;

  using iterator = stack_vector<ParallelDim, MAX_TENSOR_DIM>::iterator;
  using const_iterator =
      stack_vector<ParallelDim, MAX_TENSOR_DIM>::const_iterator;

  iterator begin();
  const_iterator begin() const;
  const_iterator cbegin() const;
  iterator end();
  const_iterator end() const;
  const_iterator cend() const;

public:
  DataType data_type;
  stack_vector<ParallelDim, MAX_TENSOR_DIM> dims;
};

std::ostream &operator<<(std::ostream &, LegionParallelTensorShape const &);
} // namespace FlexFlow

namespace std {

template <>
struct hash<::FlexFlow::LegionParallelTensorShape> {
  size_t operator()(::FlexFlow::LegionParallelTensorShape const &) const;
};

} // namespace std

#endif
