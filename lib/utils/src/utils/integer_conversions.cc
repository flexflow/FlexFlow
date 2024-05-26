#include "utils/integer_conversions.h"
#include <cassert>

namespace FlexFlow {

size_t size_t_from_int(int x) {
  assert(x >= 0);
  return static_cast<size_t>(x);
}

} // namespace FlexFlow
