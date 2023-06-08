#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_DEVICE_ID_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_DEVICE_ID_H

#include "device_type.h"
#include "utils/strong_typedef.h"
#include "utils/variant.h"

namespace FlexFlow {

struct gpu_id_t : strong_typedef<gpu_id_t, int> {
  using strong_typedef::strong_typedef;
};

struct cpu_id_t : strong_typedef<cpu_id_t, int> {
  using strong_typedef::strong_typedef;
};

using device_id_t = variant<gpu_id_t, cpu_id_t>;
device_id_t operator+(device_id_t, size_t);

DeviceType get_device_type(device_id_t);
gpu_id_t unwrap_gpu(device_id_t);
cpu_id_t unwrap_cpu(device_id_t);

device_id_t from_index(int, DeviceType);

} // namespace FlexFlow

MAKE_TYPEDEF_HASHABLE(::FlexFlow::gpu_id_t);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::gpu_id_t, "gpu_id");

MAKE_TYPEDEF_HASHABLE(::FlexFlow::cpu_id_t);
MAKE_TYPEDEF_PRINTABLE(::FlexFlow::cpu_id_t, "cpu_id");

#endif
