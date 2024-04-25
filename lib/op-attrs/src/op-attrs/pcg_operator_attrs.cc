#include "op-attrs/pcg_operator_attrs.h"

namespace FlexFlow {

bool is_parallel_op(PCGOperatorAttrs const &attrs) {
  return ( attrs.has<CombineAttrs>()
           || attrs.has<ReductionAttrs>()
           || attrs.has<RepartitionAttrs>()
           || attrs.has<ReplicateAttrs>());
}

} // namespace FlexFlow
