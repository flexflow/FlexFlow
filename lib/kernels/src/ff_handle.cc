#include "kernels/ff_handle.h"

namespace FlexFlow {

std::string format_as(PerDeviceFFHandle const &x) {
  std::ostringstream oss;
  oss << "<PerDeviceFFHandle";
  oss << " workSpaceSize=" << x.workSpaceSize;
  oss << " allowTensorOpMathConversion=" << x.allowTensorOpMathConversion;
  return oss.str();
}

std::ostream &operator<<(std::ostream &s, PerDeviceFFHandle const &x) {
  return s << fmt::to_string(x);
}

} // namespace FlexFlow
