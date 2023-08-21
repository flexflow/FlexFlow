#include "op-attrs/ops/gather.h"

namespace FlexFlow {

bool GatherAttrs::is_valid(ParallelTensorShape const &lhs,
                           ParallelTensorShape const &rhs) const {
  if (lhs.dims.num_dims() != rhs.dims.num_dims()) {
    return false;
  }
  for (auto i : lhs.dims) {
    if (ff_dim_t(i.size) != this->dim &&
        lhs.at(ff_dim_t(i.size)).size < rhs.at(ff_dim_t(i.size)).size) {
      return false;
    }
  }
  return true;
}

} // namespace FlexFlow
