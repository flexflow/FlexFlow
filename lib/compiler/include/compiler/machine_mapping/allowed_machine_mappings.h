#ifndef _FLEXFLOW_ALLOWED_MACHINE_MAPPINGS_H_
#define _FLEXFLOW_ALLOWED_MACHINE_MAPPINGS_H_

#include "compiler/machine_mapping/machine_mapping_context.dtg.h"
#include "pcg/machine_specification.dtg.h"

namespace FlexFlow {

std::vector<std::unordered_map<Node, MachineView>>
    allowed_machine_mappings(MachineMappingContext const &context,
                             std::unordered_set<Node> const &nodes,
                             MachineSpecification const &resource);

std::vector<std::unordered_map<DataflowOutput, MachineView>>
    allowed_machine_mappings(MachineMappingContext const &context,
                             std::unordered_set<DataflowOutput> const &values,
                             MachineSpecification const &resource);

}

#endif