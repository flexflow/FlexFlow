#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_COMPUTATION_GRAPH_H

#include "layer.h"
#include "operator_guid_t.h"
#include "tensor.h"
#include "utils/graph.h"
#include "utils/strong_typedef.h"
#include "visit_struct/visit_struct.hpp"

namespace FlexFlow {

struct ComputationGraph
    : public strong_typedef<ComputationGraph,
                            OutputLabelledMultiDiGraph<Layer, Tensor>> {
  using strong_typedef::strong_typedef;
};

} // namespace FlexFlow

namespace FlexFlow {
static_assert(is_well_behaved_value_type_no_hash<ComputationGraph>::value, "");
}

#endif
