#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_DEVICE_ID_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_DEVICE_ID_H

#include "pcg/cpu_id_t.dtg.h"
#include "pcg/device_id_t.dtg.h"
#include "pcg/device_type.dtg.h"
#include "pcg/gpu_id_t.dtg.h"

namespace FlexFlow {

device_id_t operator+(device_id_t, size_t);

DeviceType get_device_type(device_id_t const &device_id);
gpu_id_t unwrap_gpu(device_id_t);
cpu_id_t unwrap_cpu(device_id_t);
int get_raw_id(device_id_t);

device_id_t device_id_from_index(int, DeviceType);

} // namespace FlexFlow

#endif
