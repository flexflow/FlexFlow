#include "local-execution/device_specific_device_states.dtg.h"
#include "local-execution/task_argument_accessor.h"

namespace FlexFlow {
  
struct InitTaskImplFunction {

DeviceSpecificDeviceStates (*init_task_impl_function)(TaskArgumentAccessor const &);

bool operator==(InitTaskImplFunction const &) const;
bool operator!=(InitTaskImplFunction const &) const;

};

std::string format_as(InitTaskImplFunction const &x);
std::ostream &operator<<(std::ostream &s, InitTaskImplFunction const &x);

} // namespace FlexFlow

