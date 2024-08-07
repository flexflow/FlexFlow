#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H

#include "pcg/cpu_id_t.dtg.h"
#include "pcg/device_coordinates.dtg.h"
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

std::unordered_multiset<device_id_t> get_device_ids(MachineView const &mv);
device_id_t get_last_device_id(MachineView const &mv);

size_t num_dims(MachineView const &mv);
size_t num_devices(MachineView const &mv);
size_t get_size(MachineView const &mv);
std::unordered_multiset<num_points_t>
    get_num_devices_per_dim(MachineView const &mv);
std::unordered_multiset<side_size_t>
    get_side_size_per_dim(MachineView const &mv);

DeviceType get_device_type(MachineView const &mv);

MachineView make_1d_machine_view(gpu_id_t start, gpu_id_t stop, int stride = 1);
MachineView make_1d_machine_view(cpu_id_t start, cpu_id_t stop, int stride = 1);
MachineView
    make_1d_machine_view(device_id_t start, device_id_t stop, int stride = 1);

MachineView make_1d_machine_view(cpu_id_t start,
                                 num_points_t num_points,
                                 int stride = 1);
MachineView make_1d_machine_view(gpu_id_t start,
                                 num_points_t num_points,
                                 int stride = 1);
MachineView make_1d_machine_view(device_id_t start,
                                 num_points_t num_points,
                                 int stride = 1);

MachineView make_1d_machine_view(cpu_id_t start,
                                 side_size_t interval_size,
                                 int stride = 1);
MachineView make_1d_machine_view(gpu_id_t start,
                                 side_size_t interval_size,
                                 int stride = 1);
MachineView make_1d_machine_view(device_id_t start,
                                 side_size_t interval_size,
                                 int stride = 1);

} // namespace FlexFlow

#endif
