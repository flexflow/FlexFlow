#include "kernels/array_shape.h"
#include "utils/containers/product.h"

namespace FlexFlow {

static LegionTensorDims
    legion_dims_from_ff_dims(FFOrdered<size_t> const &ff_ordered) {
  std::vector<size_t> sizes(ff_ordered.size());
  std::reverse_copy(ff_ordered.begin(), ff_ordered.end(), sizes.begin());
  return LegionTensorDims(sizes.begin(), sizes.end());
}

ArrayShape::ArrayShape(size_t *_dims, size_t num_dims)
    : dims(_dims, _dims + num_dims) {}

ArrayShape::ArrayShape(TensorShape const &shape)
    : dims(legion_dims_from_ff_dims(shape.dims.ff_ordered)) {}

ArrayShape::ArrayShape(std::vector<std::size_t> const &input_dims)
    : dims(input_dims) {}

std::size_t ArrayShape::get_volume() const {
  return this->num_elements();
}

std::size_t ArrayShape::num_dims() const {
  return this->dims.size();
}

std::size_t ArrayShape::get_dim() const {
  return this->num_dims();
}

std::size_t ArrayShape::num_elements() const {
  if (dims.size() == 0) {
    return 0;
  }
  return product(this->dims);
}

std::size_t ArrayShape::operator[](legion_dim_t idx) const {
  return dims.at(idx);
}

std::size_t ArrayShape::at(legion_dim_t idx) const {
  return dims.at(idx);
}

std::size_t ArrayShape::at(ff_dim_t idx) const {
  return dims.at(legion_dim_from_ff_dim(idx, this->num_dims()));
}

ArrayShape ArrayShape::sub_shape(
    std::optional<std::variant<ff_dim_t, legion_dim_t>> start,
    std::optional<std::variant<ff_dim_t, legion_dim_t>> end) const {

  NOT_IMPLEMENTED();
}

std::optional<std::size_t> ArrayShape::at_maybe(legion_dim_t index) const {
  if (index.value < dims.size()) {
    return dims.at(index);
  } else {
    return std::nullopt;
  }
}

std::optional<std::size_t> ArrayShape::at_maybe(ff_dim_t index) const {
  return this->at_maybe(legion_dim_from_ff_dim(index, this->num_dims()));
}

size_t get_volume(ArrayShape const &shape) {
  return shape.get_volume();
}

TensorShape get_tensor_shape(ArrayShape const &shape, DataType dtype) {
  return TensorShape{TensorDims{ff_ordered_from_legion_ordered(shape.dims)},
                     dtype};
}

std::string format_as(ArrayShape const &x) {
  std::ostringstream oss;
  oss << "<ArrayShape";
  oss << " dims=" << x.dims;
  oss << ">";
  return oss.str();
}

std::ostream &operator<<(std::ostream &s, ArrayShape const &x) {
  return (s << fmt::to_string(x));
}

} // namespace FlexFlow
