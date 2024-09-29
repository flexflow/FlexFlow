#include "compiler/machine_mapping/machine_mapping_problem_tree.h"
#include "compiler/machine_mapping/full_binary_tree/get_left_child.h"
#include "compiler/machine_mapping/full_binary_tree/get_right_child.h"
#include "compiler/machine_mapping/full_binary_tree/require.h"
#include "compiler/machine_mapping/full_binary_tree/visit.h"
#include "compiler/machine_mapping/full_binary_tree/get_leaves.h"
#include "utils/overload.h"
#include "compiler/machine_mapping/mm_problem_tree_split_label.h"

namespace FlexFlow {

MachineMappingProblemTree mm_problem_tree_make_series_split(AbstractedTensorSetMovement const &tensor_set_movement,
                                                            MachineMappingProblemTree const &lhs, 
                                                            MachineMappingProblemTree const &rhs) {
  return MachineMappingProblemTree{
    FullBinaryTree<MMProblemTreeSplitLabel, PCGOperatorAttrs>{
      FullBinaryTreeParentNode<MMProblemTreeSplitLabel, PCGOperatorAttrs>{
        /*label=*/MMProblemTreeSplitLabel{
          MMProblemTreeSeriesSplitLabel{
            /*tensor_set_movement=*/tensor_set_movement,
          },
        },
        /*lhs=*/lhs.raw_tree,
        /*rhs=*/rhs.raw_tree,
      },
    },
  };
}

MachineMappingProblemTree mm_problem_tree_make_parallel_split(MachineMappingProblemTree const &lhs,
                                                              MachineMappingProblemTree const &rhs) {
  return MachineMappingProblemTree{
    FullBinaryTree<MMProblemTreeSplitLabel, PCGOperatorAttrs>{
      FullBinaryTreeParentNode<MMProblemTreeSplitLabel, PCGOperatorAttrs>{
        /*label=*/MMProblemTreeSplitLabel{
          MMProblemTreeParallelSplitLabel{},
        },
        /*lhs=*/lhs.raw_tree,
        /*rhs=*/rhs.raw_tree,
      },
    },
  };
}

MachineMappingProblemTree mm_problem_tree_make_leaf(PCGOperatorAttrs const &layer) {
  return MachineMappingProblemTree{
    FullBinaryTree<MMProblemTreeSplitLabel, PCGOperatorAttrs>{
      layer,
    },
  };
}

SPDecompositionTreeNodeType get_node_type(MachineMappingProblemTree const &tree) {
  return visit<SPDecompositionTreeNodeType>(
    tree.raw_tree,
    overload {
      [](FullBinaryTreeParentNode<MMProblemTreeSplitLabel, PCGOperatorAttrs> const &parent) {
        return split_label_get_node_type(parent.label);
      },
      [](PCGOperatorAttrs const &) {
        return SPDecompositionTreeNodeType::NODE;
      }
    });
}


MMProblemTreeSeriesSplit require_series_split(MachineMappingProblemTree const &t) {
  FullBinaryTreeParentNode<MMProblemTreeSplitLabel, PCGOperatorAttrs> raw_node = require_parent_node(t.raw_tree);

  return MMProblemTreeSeriesSplit{
    /*label=*/raw_node.label.get<MMProblemTreeSeriesSplitLabel>(),
    /*left=*/MachineMappingProblemTree{get_left_child(raw_node)},
    /*right=*/MachineMappingProblemTree{get_right_child(raw_node)},
  };
}

MMProblemTreeParallelSplit require_parallel_split(MachineMappingProblemTree const &t) {
  FullBinaryTreeParentNode<MMProblemTreeSplitLabel, PCGOperatorAttrs> raw_node = require_parent_node(t.raw_tree);

  return MMProblemTreeParallelSplit{
    /*label=*/raw_node.label.get<MMProblemTreeParallelSplitLabel>(),
    /*left=*/MachineMappingProblemTree{get_left_child(raw_node)},
    /*right=*/MachineMappingProblemTree{get_right_child(raw_node)},
  };
}

PCGOperatorAttrs require_leaf(MachineMappingProblemTree const &t) {
  return require_leaf(t.raw_tree);
}

std::unordered_multiset<PCGOperatorAttrs> get_leaves(MachineMappingProblemTree const &t) {
  return get_leaves(t.raw_tree);
}

} // namespace FlexFlow
