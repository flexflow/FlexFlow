#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/merge_maps.h"
#include "utils/containers/try_at.h"
#include "utils/full_binary_tree/binary_tree_path.h"

namespace FlexFlow {

ParallelLayerGuidObliviousMachineMapping binary_combine_mappings(
    ParallelLayerGuidObliviousMachineMapping const &lhs,
    ParallelLayerGuidObliviousMachineMapping const &rhs) {
  return ParallelLayerGuidObliviousMachineMapping{
      merge_maps(map_keys(lhs.raw_mapping, nest_inside_left_child),
                 map_keys(rhs.raw_mapping, nest_inside_right_child)),
  };
}

std::optional<MachineView> get_machine_view_for_path(
    ParallelLayerGuidObliviousMachineMapping const &mapping,
    BinaryTreePath const &path) {
  return try_at(mapping.raw_mapping, path);
}

} // namespace FlexFlow
