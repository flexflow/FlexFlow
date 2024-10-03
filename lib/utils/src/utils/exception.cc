#include "utils/exception.h"

namespace FlexFlow {

std::runtime_error mk_runtime_error(std::string const &s) {
  return std::runtime_error(s);
}

} // namespace FlexFlow
