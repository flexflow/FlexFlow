#ifndef _FLEXFLOW_KERNELS_ARRAY_SHAPE_H
#define _FLEXFLOW_KERNELS_ARRAY_SHAPE_H

#include "legion_dim.h"
#include "op-attrs/tensor_shape.h"
#include "utils/stack_vector.h"
#include "utils/visitable.h"
#include <cstddef>
#include <vector>

namespace FlexFlow {

struct ArrayShape {
public:
  ArrayShape() = delete;
  ArrayShape(size_t *dims, size_t num_dims);
  ArrayShape(TensorShape const &shape);
  ArrayShape(std::vector<std::size_t> const &);

  /**
   * @brief Alias of ArrayShape::num_elements for compatibility with
   * Legion::Domain
   */
  std::size_t get_volume() const;

  /**
   * @brief Alias of ArrayShape::num_dims for compatibility with Legion::Domain
   */
  std::size_t get_dim() const;

  std::size_t num_elements() const;
  std::size_t num_dims() const;

  std::size_t operator[](legion_dim_t) const;
  std::size_t at(legion_dim_t) const;
  std::size_t at(ff_dim_t) const;

  legion_dim_t last_idx() const;
  legion_dim_t neg_idx(int) const;

  std::optional<std::size_t> at_maybe(std::size_t) const;

  ArrayShape reversed_dim_order() const;
  ArrayShape sub_shape(std::optional<legion_dim_t> start,
                       std::optional<legion_dim_t> end) const;

public:
  LegionTensorDims dims;
};
FF_VISITABLE_STRUCT_NONSTANDARD_CONSTRUCTION(ArrayShape, dims);

size_t get_volume(ArrayShape const &);

} // namespace FlexFlow

#endif
