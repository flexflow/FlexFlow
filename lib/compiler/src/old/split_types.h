#ifndef _FLEXFLOW_FFC_SPLIT_TYPES_H
#define _FLEXFLOW_FFC_SPLIT_TYPES_H

#include "node.h"
#include "pcg/machine_view.h"

namespace FlexFlow {
namespace PCG {

enum class SplitType { SEQUENTIAL, VERTICAL, HORIZONTAL };

struct NonsequenceSplit {
  SplitType type;
  int param;
  bool flip_graphs;

  static NonsequenceSplit sequential();
  static NonsequenceSplit vertical(int param, bool flip_graphs);
  static NonsequenceSplit horizontal(int param, bool flip_graphs);
};

struct NodeAssignment {
  Node node;
  MachineView view;
};

using SequenceSplit = NodeAssignment;

} // namespace PCG
} // namespace FlexFlow

#endif
