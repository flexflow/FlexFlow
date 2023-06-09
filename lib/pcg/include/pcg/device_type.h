#ifndef _FLEXFLOW_PCG_INCLUDE_PCG_DEVICE_TYPE_H
#define _FLEXFLOW_PCG_INCLUDE_PCG_DEVICE_TYPE_H

#include "utils/fmt.h"

namespace FlexFlow {

enum class DeviceType { GPU, CPU };

}

namespace fmt {

template <>
struct formatter<::FlexFlow::DeviceType> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::DeviceType d, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using ::FlexFlow::DeviceType;

    string_view name = "unknown";
    switch (d) {
      case DeviceType::GPU:
        name = "GPU";
        break;
      case DeviceType::CPU:
        name = "CPU";
        break;
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt

#endif
