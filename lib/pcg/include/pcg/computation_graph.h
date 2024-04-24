#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H

#include "pcg/computation_graph.dtg.h"
#include "pcg/tensor_guid_t.dtg.h"
#include "pcg/tensor_attrs.dtg.h"

namespace FlexFlow {

TensorAttrs get_tensor_attrs(ComputationGraph const &, tensor_guid_t const &);

} // namespace FlexFlow

#endif
