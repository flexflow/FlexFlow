#include "op-attrs/ops/groupby.h"
#include "utils/exception.decl.h"

namespace FlexFlow {

bool Group_byAttrs::is_valid(ParallelTensorShape const &lhs,
                             ParallelTensorShape const &rhs) const {
  if (!lhs.is_valid() || !rhs.is_valid()) {
    return false;
  }
  NOT_IMPLEMENTED();
}

ParallelTensorShape get_output_shape(Group_byAttrs const &attrs,
                                     ParallelTensorShape const &lhs,
                                     ParallelTensorShape const &rhs) {
  NOT_IMPLEMENTED();
}

} // namespace FlexFlow
