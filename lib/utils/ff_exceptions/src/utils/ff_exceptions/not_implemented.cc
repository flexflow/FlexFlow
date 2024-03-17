#include "utils/ff_exceptions/not_implemented.h"

namespace FlexFlow {

not_implemented::not_implemented() : std::logic_error("not implemented") {}

} // namespace FlexFlow
