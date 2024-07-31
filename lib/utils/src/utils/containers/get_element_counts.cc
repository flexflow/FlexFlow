#include "utils/containers/get_element_counts.h"
#include "utils/containers/as_vector.h"

namespace FlexFlow {

std::unordered_map<char, int> get_element_counts(std::string const &s) {
  return get_element_counts(as_vector(s));
}

} // namespace FlexFlow
