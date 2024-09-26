#ifndef _FLEXFLOW_ALLOWED_MACHINE_MAPPINGS_H_
#define _FLEXFLOW_ALLOWED_MACHINE_MAPPINGS_H_

#include "compiler/machine_mapping/machine_mapping_context.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/parallel_computation_graph/parallel_layer_guid_t.dtg.h"
#include "pcg/parallel_computation_graph/parallel_tensor_guid_t.dtg.h"
#include <vector>

namespace FlexFlow {

std::vector<std::unordered_map<parallel_layer_guid_t, MachineView>>
    get_allowed_machine_views_list(
        MachineMappingContext const &context,
        std::unordered_set<parallel_layer_guid_t> const &layers,
        MachineSpecification const &resource);

std::vector<std::unordered_map<parallel_tensor_guid_t, MachineView>>
    get_allowed_src_machine_views_list(
        MachineMappingContext const &context,
        std::unordered_set<parallel_tensor_guid_t> const &values,
        MachineSpecification const &resource);

} // namespace FlexFlow

#endif
