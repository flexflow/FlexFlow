#ifndef _FLEXFLOW_RUNTIME_SRC_DEVICE_SPECIFIC_ARG_H
#define _FLEXFLOW_RUNTIME_SRC_DEVICE_SPECIFIC_ARG_H

#include "serialization.h"
#include "task_argument_accessor.h"
#include "utils/exception.h"

namespace FlexFlow {

/**
 * \class DeviceSpecificArg
 * \brief Holds a ptr and device index
 * 
 * An argument slot can be used to pass small (not tensor-sized) values of arbitrary (either be serializable or *device-specific*) types via Legion::TaskArgument
*/
template <typename T>
struct DeviceSpecificArg {
  /**
   * \fn T *get(TaskArgumentAccessor const &accessor) const
   * \brief Checks that device index is correct and throw error if not
   * \param accessor TaskArgumentAccessor used to get device index
  */
  T *get(TaskArgumentAccessor const &accessor) const {
    if (accessor.get_device_idx() != this->device_idx) {
      throw mk_runtime_error("Invalid access to DeviceSpecificArg: attempted "
                             "device_idx {} != correct device_idx {})",
                             accessor.get_device_idx(),
                             this->device_idx);
    }
  }

private:
  T *ptr;
  size_t device_idx;
};

/**
 * \class is_trivially_serializable
 * \brief manually force serialization to make DeviceSpecificArgs trivially
 * 
 * Serializable; todo add more details
*/
template <typename T>
struct is_trivially_serializable<DeviceSpecificArg<T>> : std::true_type {};

}

#endif
