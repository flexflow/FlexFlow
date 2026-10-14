#include "compiler/machine_mapping/parallel_layer_guid_oblivious_machine_mapping.h"
#include "compiler/machine_mapping/machine_mapping_problem_tree/machine_mapping_problem_tree.h"
#include "compiler/series_parallel/pcg/pcg_binary_sp_decomposition.h"
#include "op-attrs/computation_graph_op_attrs.h"
#include "op-attrs/get_operator_task_space.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "pcg/parallel_computation_graph/parallel_computation_graph.h"
#include "utils/containers/binary_merge_disjoint_maps.h"
#include "utils/containers/map_keys.h"
#include "utils/containers/require_same.h"
#include "utils/containers/try_at.h"
#include "utils/full_binary_tree/binary_tree_path.h"
#include "utils/not_implemented.h"

namespace FlexFlow {

ParallelLayerGuidObliviousMachineMapping binary_combine_mappings(
    ParallelLayerGuidObliviousMachineMapping const &lhs,
    ParallelLayerGuidObliviousMachineMapping const &rhs) {
  return ParallelLayerGuidObliviousMachineMapping{
      binary_merge_disjoint_maps(
          map_keys(lhs.raw_mapping, nest_inside_left_child),
          map_keys(rhs.raw_mapping, nest_inside_right_child)),
  };
}

ParallelLayerGuidObliviousMachineMapping
    restrict_to_left_child(ParallelLayerGuidObliviousMachineMapping const &) {
  NOT_IMPLEMENTED();
}

ParallelLayerGuidObliviousMachineMapping
    restrict_to_right_child(ParallelLayerGuidObliviousMachineMapping const &) {
  NOT_IMPLEMENTED();
}

std::optional<MachineView> get_machine_view_for_path(
    ParallelLayerGuidObliviousMachineMapping const &mapping,
    BinaryTreePath const &path) {
  return try_at(mapping.raw_mapping, path);
}

std::map<BinaryTreePath, MachineSpaceStencil>
    get_machine_stencils_for_decomposition(
        ParallelComputationGraph const &pcg,
        PCGBinarySPDecomposition const &decomposition,
        ParallelLayerGuidObliviousMachineMapping const &mapping) {
  std::set<BinaryTreePath> leaf_paths = require_same(
      pcg_sp_tree_get_all_leaf_paths(decomposition), keys(mapping.raw_mapping));

  std::map<BinaryTreePath, OperatorTaskSpace> path_to_op_task_space_map =
      map_values(pcg_sp_tree_get_path_to_leaf_map(decomposition),
                 [&](parallel_layer_guid_t l) -> OperatorTaskSpace {
                   return get_operator_task_space(pcg, l);
                 });

  return generate_map(
      leaf_paths, [&](BinaryTreePath const &p) -> MachineSpaceStencil {
        return MachineSpaceStencil{
            /*operator_task_space=*/path_to_op_task_space_map.at(p),
            /*machine_view=*/mapping.raw_mapping.at(p),
        };
      });
}

std::map<BinaryTreePath, std::optional<MachineSpaceStencil>>
    get_machine_stencils_for_mm_problem_tree(
        MachineMappingProblemTree const &tree,
        ParallelLayerGuidObliviousMachineMapping const &mapping) {

  std::map<BinaryTreePath, UnmappedRuntimeOnlyOpCostEstimateKey> tree_leaf_map =
      mm_problem_tree_get_path_to_leaf_map(tree);

  std::set<BinaryTreePath> mapping_paths = keys(mapping.raw_mapping);
  std::set<BinaryTreePath> tree_paths = keys(tree_leaf_map);

  ASSERT(is_subseteq_of(mapping_paths, tree_paths));

  return generate_map(
      tree_paths,
      [&](BinaryTreePath const &p) -> std::optional<MachineSpaceStencil> {
        if (!contains_key(mapping.raw_mapping, p)) {
          return std::nullopt;
        }

        UnmappedRuntimeOnlyOpCostEstimateKey leaf = tree_leaf_map.at(p);

        ComputationGraphOpAttrs leaf_op_attrs =
            compgraph_op_attrs_from_pcg_op_attrs(leaf.op_attrs).value();

        std::map<TensorSlotName, ParallelTensorDimDegrees> leaf_input_degrees =
            map_values(leaf.input_shapes, [](ParallelTensorShape const &s) {
              return get_parallel_degrees(s);
            });

        return MachineSpaceStencil{
            /*operator_task_space=*/get_operator_task_space(leaf_op_attrs,
                                                            leaf_input_degrees),
            /*machine_view=*/mapping.raw_mapping.at(p),
        };
      });
}

std::map<BinaryTreePath, MachineSpaceStencil>
    get_machine_stencils_for_partially_mapped_mm_problem_tree(
        MachineMappingProblemTree const &tree,
        ParallelLayerGuidObliviousMachineMapping const &mappings) {

  return filtermap_values(
      get_machine_stencils_for_mm_problem_tree(tree, mappings),
      [](std::optional<MachineSpaceStencil> const &s) { return s; });
}

} // namespace FlexFlow
