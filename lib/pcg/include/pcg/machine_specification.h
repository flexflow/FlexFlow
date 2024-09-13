#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_SPECIFICATION_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_SPECIFICATION_H

#include "pcg/device_id_t.dtg.h"
#include "pcg/device_type.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification_coordinate.dtg.h"

namespace FlexFlow {

int get_num_gpus(MachineSpecification const &ms);
int get_num_cpus(MachineSpecification const &ms);
int get_num_devices(MachineSpecification const &ms,
                    DeviceType const &device_type);
int get_num_devices_per_node(MachineSpecification const &ms,
                             DeviceType const &device_type);

bool is_valid_machine_specification_coordinates(
    MachineSpecification const &ms,
    MachineSpecificationCoordinate const &coord);

device_id_t get_device_id(MachineSpecification const &ms,
                          MachineSpecificationCoordinate const &coord);
} // namespace FlexFlow

#endif
