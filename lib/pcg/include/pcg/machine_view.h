#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H

#include "pcg/cpu_id_t.dtg.h"
#include "pcg/device_id.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/device_type.dtg.h"
#include "pcg/gpu_id_t.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/num_points_t.dtg.h"
#include "pcg/side_size_t.dtg.h"
#include <cstddef>
#include <vector>

namespace FlexFlow {

std::vector<device_id_t> device_ids(MachineView const &);
size_t num_dims(MachineView const &);
std::size_t num_devices(MachineView const &);
DeviceType get_device_type(MachineView const &);

MachineView make_1d_machine_view(gpu_id_t start, gpu_id_t stop, int stride = 1);
MachineView make_1d_machine_view(cpu_id_t start, cpu_id_t stop, int stride = 1);
MachineView
    make_1d_machine_view(device_id_t start, device_id_t stop, int stride = 1);

MachineView make_1d_machine_view(gpu_id_t start,
                                 num_points_t num_points,
                                 int stride = 1);
MachineView make_1d_machine_view(cpu_id_t start,
                                 num_points_t num_points,
                                 int stride = 1);
MachineView make_1d_machine_view(device_id_t start,
                                 num_points_t num_points,
                                 int stride = 1);

MachineView make_1d_machine_view(gpu_id_t start,
                                 side_size_t interval_size,
                                 int stride = 1);
MachineView make_1d_machine_view(cpu_id_t start,
                                 side_size_t interval_size,
                                 int stride = 1);
MachineView make_1d_machine_view(device_id_t start,
                                 side_size_t interval_size,
                                 int stride = 1);

MachineView make_1d_machine_view(device_id_t start, size_t interval_size);

} // namespace FlexFlow

#endif
