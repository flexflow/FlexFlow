#include "utils/containers/get_element_counts.h"
#include "utils/containers/vector_of.h"

namespace FlexFlow {

std::unordered_map<char, int> get_element_counts(std::string const &s) {
  return get_element_counts(vector_of(s));
}

} // namespace FlexFlow
