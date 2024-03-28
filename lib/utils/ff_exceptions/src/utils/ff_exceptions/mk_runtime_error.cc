#include "utils/ff_exceptions/mk_runtime_error.h"

namespace FlexFlow {

std::runtime_error mk_runtime_error(std::string const &msg) {
  return std::runtime_error{msg};
}

} // namespace FlexFlow
