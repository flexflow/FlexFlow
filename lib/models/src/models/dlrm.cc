#include "models/dlrm.h"
#include "pcg/computation_graph.h"

namespace FlexFlow {

DLRMConfig get_default_dlrm_config() {
  return DLRMConfig(64);
}

ComputationGraph get_dlrm_computation_graph(DLRMConfig const &config) {
  ComputationGraphBuilder cgb;
  return cgb.computation_graph;
}

} // namespace FlexFlow
