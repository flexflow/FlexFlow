#include "utils/containers/count.h"

namespace FlexFlow {

std::vector<size_t> count(size_t n) {
  std::vector<size_t> v(n);
  for (size_t i = 0; i < n; i++) {
    v[i] = i;
  }
  return v;
}

} // namespace FlexFlow
