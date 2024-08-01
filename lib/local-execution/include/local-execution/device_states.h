#include "local-execution/device_specific_device_states.dtg.h"
#include "local-execution/device_states.dtg.h"

namespace FlexFlow {

DeviceStates
    get_device_state_from_device_specific(DeviceSpecificDeviceStates const &,
                                          size_t device_idx);

}
