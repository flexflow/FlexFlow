#ifndef _FLEXFLOW_COMPILER_ALLOWED_MACHINE_VIEWS_H
#define _FLEXFLOW_COMPILER_ALLOWED_MACHINE_VIEWS_H

#include "compiler/machine_view_to_tensor_mapping.dtg.h"
#include "op-attrs/parallel_tensor_shape.dtg.h"
#include "pcg/machine_specification.h"
#include "pcg/machine_view.h"
#include "pcg/machine_view_projection.dtg.h"
#include "pcg/start_invariant_machine_view.dtg.h"

namespace FlexFlow {

bool is_valid_machine_view(MachineView const &mv,
                           MachineSpecification const &machine_spec);

bool is_valid_machine_view(MachineView const &mv,
                           ParallelTensorShape const &shape);

std::unordered_set<std::pair<MachineView, MachineViewProjection>>
    get_allowed_partial_machine_view_mappings(
        MachineSpecification const &machine_spec,
        ParallelTensorShape const &shape,
        DeviceType device_type = DeviceType::GPU);
std::unordered_set<std::pair<StartInvariantMachineView, MachineViewProjection>>
    get_allowed_partial_start_invariant_machine_view_mappings(
        MachineSpecification const &machine_spec,
        ParallelTensorShape const &shape,
        DeviceType device_type);
} // namespace FlexFlow

#endif
