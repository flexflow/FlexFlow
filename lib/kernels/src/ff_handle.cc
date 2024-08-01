#include "kernels/ff_handle.h"

namespace FlexFlow {

std::string format_as(PerDeviceFFHandle const &x) {
  return fmt::format("PerDeviceFFHandle");
}

std::ostream &operator<<(std::ostream &s, PerDeviceFFHandle const &x) {
  return (s << fmt::to_string(x));
}

}
