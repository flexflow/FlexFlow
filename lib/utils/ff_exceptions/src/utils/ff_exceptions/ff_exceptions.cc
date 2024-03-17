#include "utils/ff_exceptions/ff_exceptions.h"

namespace FlexFlow {

std::runtime_error mk_runtime_error(std::string const &msg) {
  return std::runtime_error{msg};
}

} // namespace FlexFlow
