#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_SPECIFICATION_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_SPECIFICATION_H

#include "pcg/device_type.dtg.h"
#include "pcg/machine_specification.dtg.h"

namespace FlexFlow {

int get_num_gpus(MachineSpecification const &ms);
int get_num_cpus(MachineSpecification const &ms);
int get_num_devices(MachineSpecification const &ms,
                    DeviceType const &device_type);

} // namespace FlexFlow

#endif
