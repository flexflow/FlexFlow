#include "pcg/parallel_computation_graph/parallel_layer_attrs.h"
#include "op-attrs/pcg_operator_attrs.h"

namespace FlexFlow {

OperatorType get_op_type(ParallelLayerAttrs const &a) {
  return get_op_type(a.op_attrs);
}

} // namespace FlexFlow
