#include "local-execution/device_specific_device_states.dtg.h"
#include "local-execution/per_device_state.dtg.h"

namespace FlexFlow {

PerDeviceState
    get_device_state_from_device_specific(DeviceSpecificDeviceStates const &,
                                          size_t device_idx);

}
