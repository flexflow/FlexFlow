#ifndef _FLEXFLOW_COMPILER_COMPILER_H
#define _FLEXFLOW_COMPILER_COMPILER_H

#include "pcg/machine_view.h"

namespace FlexFlow {

struct SearchSolution {
  ParallelComputationGraph optimized_pcg;
  std::unordered_map<Node, MachineView> device_assignments;
};

SearchSolution run_data_parallelize(ComputationGraph const &, MachineSpecification const &);

}

#endif
