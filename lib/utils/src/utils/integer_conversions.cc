#include "utils/integer_conversions.h"
#include <cassert>
#include <limits>

namespace FlexFlow {

size_t size_t_from_int(int x) {
  assert(x >= 0);
  return static_cast<size_t>(x);
}

int int_from_size_t(size_t x) {
  assert (x < std::numeric_limits<int>::max());
  return static_cast<int>(x);
}

} // namespace FlexFlow
