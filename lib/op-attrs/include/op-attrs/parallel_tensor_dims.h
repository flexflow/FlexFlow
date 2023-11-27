#ifndef _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIMS_H
#define _FLEXFLOW_OP_ATTRS_INCLUDE_OP_ATTRS_PARALLEL_TENSOR_DIMS_H

#include "parallel_dim.h"
#include "tensor_shape.h"
#include "utils/visitable.h"

namespace FlexFlow {

struct ParallelTensorDims {
  explicit ParallelTensorDims(TensorDims const &);

  template <typename Dims>
  ParallelTensorDims(std::vector<Dims> const &dims) : data(dims) {}

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

  iterator begin();
  const_iterator begin() const;
  const_iterator cbegin() const;
  iterator end();
  const_iterator end() const;
  const_iterator cend() const;
  reverse_iterator rbegin();
  const_reverse_iterator rbegin() const;
  const_reverse_iterator crbegin() const;
  reverse_iterator rend();
  const_reverse_iterator rend() const;
  const_reverse_iterator crend() const;

public:
  req<FFOrdered<ParallelDim>> data;
};

FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(ParallelTensorDims, data);

bool is_valid(ParallelTensorDims const &);
TensorDims get_piece_dims(ParallelTensorDims const &);
TensorDims get_tensor_dims_unsafe(ParallelTensorDims const &);

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::ParallelTensorDims> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::ParallelTensorDims dims, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using namespace FlexFlow;

    std::vector<ParallelDim> v(dims.data.begin(), dims.data.end());
    return formatter<string_view>::format(fmt::to_string(v), ctx);
  }
};

} // namespace fmt

#endif
