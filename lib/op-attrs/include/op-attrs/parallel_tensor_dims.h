#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIMS_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIMS_H

#include "parallel_dim.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ParallelTensorDims : public use_visitable_cmp<ParallelTensorDims> {
  explicit ParallelTensorDims(TensorDims const &);

  size_t get_volume() const;
  size_t num_dims() const;

  using iterator = typename FFOrdered<ParallelDim>::iterator;
  using const_iterator = typename FFOrdered<ParallelDim>::const_iterator;
  using reverse_iterator = typename FFOrdered<ParallelDim>::reverse_iterator;
  using const_reverse_iterator =
      typename FFOrdered<ParallelDim>::const_reverse_iterator;
  using value_type = typename FFOrdered<ParallelDim>::value_type;
  using pointer = typename FFOrdered<ParallelDim>::pointer;
  using const_pointer = typename FFOrdered<ParallelDim>::const_pointer;

  ParallelDim const &at(ff_dim_t const &) const;
  ParallelDim &at(ff_dim_t const &);

  iterator begin() {
    return this->data.begin();
  }

  const_iterator begin() const {
    return this->cbegin();
  };

  const_iterator cbegin() const {
    return this->data.cbegin();
  }

  iterator end() {
    return this->data.end();
  }

  const_iterator end() const {
    return this->cend();
  }

  const_iterator cend() const {
    return this->data.cend();
  }

  reverse_iterator rbegin() {
    return this->data.rbeigin();
  }

  const_reverse_iterator rbegin() const {
    return this->cbegin();
  }
  const_reverse_iterator crbegin() const {
    return this->data.crbegin();
  }

  reverse_iterator rend() {
    return this->data.rend();
  }

  const_reverse_iterator rend() const {
    return this->crend();
  }

  const_reverse_iterator crend() const {
    return this->data.crend();
  }

public:
  FFOrdered<ParallelDim> data;
};

bool is_valid(ParallelTensorDims const &);
TensorDims get_piece_dims(ParallelTensorDims const &);
TensorDims get_tensor_dims_unsafe(ParallelTensorDims const &);

} // namespace FlexFlow

VISITABLE_STRUCT(::FlexFlow::ParallelTensorDims, data);
MAKE_VISIT_HASHABLE(::FlexFlow::ParallelTensorDims);

#endif
