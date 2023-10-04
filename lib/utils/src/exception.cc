#include "utils/exception.h"
#include "utils/fmt.decl.h"

namespace FlexFlow {

not_implemented::not_implemented(char const *file, unsigned line)
    : std::logic_error(
          fmt::format("Function not yet implemented: {}:{}", file, line)) {}

not_reachable::not_reachable(char const *file, unsigned line)
    : std::logic_error(
          fmt::format("Unreachable code was reached: {}:{}", file, line)) {}

} // namespace FlexFlow
