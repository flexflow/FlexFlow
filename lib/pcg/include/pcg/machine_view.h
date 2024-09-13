#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_MACHINE_VIEW_H

#include "pcg/device_id.h"
#include "pcg/device_type.dtg.h"
#include "pcg/machine_specification.dtg.h"
#include "pcg/machine_specification_coordinates.dtg.h"
#include "pcg/machine_view.dtg.h"
#include "pcg/machine_view_coordinates.dtg.h"
#include "pcg/machine_view_dim_idx_t.dtg.h"
#include "pcg/machine_view_projection.dtg.h"
#include "pcg/num_points_t.dtg.h"
#include "pcg/side_size_t.dtg.h"
#include <cstddef>
#include <vector>

namespace FlexFlow {

std::unordered_set<MachineViewCoordinates>
    get_devices_coordinates(MachineView const &mv);
MachineViewCoordinates get_maximum_device_coordinates(MachineView const &mv);

MachineSpecificationCoordinates get_machine_specification_coordinates(
    MachineView const &mv,
    MachineViewCoordinates const &coordinates,
    MachineSpecification const &ms,
    MachineViewProjection const &projection);
StridedRectangleSide get_side_at_idx(MachineView const &mv,
                                     machine_view_dim_idx_t const &idx);

device_id_t get_device_id(MachineView const &mv,
                          MachineViewCoordinates const &coordinates,
                          MachineSpecification const &ms,
                          MachineViewProjection const &projection);
std::unordered_set<device_id_t>
    get_device_ids(MachineView const &mv,
                   MachineSpecification const &ms,
                   MachineViewProjection const &projection);

size_t num_dims(MachineView const &mv);
size_t num_devices(MachineView const &mv);
std::vector<num_points_t> get_num_devices_per_dim(MachineView const &mv);
std::vector<side_size_t> get_side_size_per_dim(MachineView const &mv);

MachineView make_1d_machine_view(int start,
                                 int stop,
                                 stride_t stride = stride_t{1},
                                 DeviceType device_type = DeviceType::GPU);

MachineView make_1d_machine_view(int start,
                                 num_points_t num_points,
                                 stride_t stride = stride_t{1},
                                 DeviceType device_type = DeviceType::GPU);

MachineView make_1d_machine_view(int start,
                                 side_size_t interval_size,
                                 stride_t stride = stride_t{1},
                                 DeviceType device_type = DeviceType::GPU);

} // namespace FlexFlow

#endif
