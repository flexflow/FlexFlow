#include "utils/exception.h"

namespace FlexFlow {

not_implemented::not_implemented()
    : std::logic_error("Function not yet implemented"){};

std::runtime_error mk_runtime_error(std::string const &msg) {
  return std::runtime_error(msg);
}

} // namespace FlexFlow
