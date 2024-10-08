#include "compiler/machine_mapping/get_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/abstracted_tensor_set_movement.h"
#include "compiler/machine_mapping/abstracted_tensor_set_movement/get_abstracted_tensor_set_movement_across_split.h"
#include "compiler/machine_mapping/transitive_reduced_pcg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.dtg.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph_edge.h"
#include "utils/containers/generate_map.h"
#include "utils/containers/keys.h"
#include "utils/containers/sum.h"
#include "utils/containers/values.h"

namespace FlexFlow {

TensorSetMovement get_tensor_set_movement_across_split(
    TransitiveReducedPCG const &tr_pcg,
    PCGBinarySeriesSplit const &split,
    ParallelLayerGuidObliviousMachineMapping const &pre_mapping,
    ParallelLayerGuidObliviousMachineMapping const &post_mapping) {
  AbstractedTensorSetMovement abstracted =
      get_abstracted_tensor_set_movement_across_split(tr_pcg, split);
  return concretize_abstracted_tensor_set_movement(
      abstracted, pre_mapping, post_mapping);
}

} // namespace FlexFlow
