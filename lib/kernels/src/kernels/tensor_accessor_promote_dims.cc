#include "kernels/tensor_accessor_promote_dims.h"
#include "op-attrs/ff_ordered/ff_ordered_filtrans.h"
#include "utils/int_ge_two/int_ge_two.h"

namespace FlexFlow {

static bool can_promote_dims_to(TensorDims const &from, TensorDims const &to) {
  auto try_require_ge_two = [](positive_int x) -> std::optional<int_ge_two> {
    if (x == 1) {
      return std::nullopt;
    } else {
      return int_ge_two{x};
    }
  };

  FFOrdered<int_ge_two> current_nontrivial_dims =
      ff_ordered_filtrans(from.ff_ordered, try_require_ge_two);

  FFOrdered<int_ge_two> new_nontrivial_dims =
      ff_ordered_filtrans(to.ff_ordered, try_require_ge_two);

  return current_nontrivial_dims == new_nontrivial_dims;
}

GenericTensorAccessorR
    tensor_accessor_promote_dims(GenericTensorAccessorR const &accessor,
                                 TensorDims const &new_dims) {
  ASSERT(can_promote_dims_to(accessor.shape.dims, new_dims));

  GenericTensorAccessorR result = accessor;
  result.shape.dims = new_dims;

  return result;
}

} // namespace FlexFlow
