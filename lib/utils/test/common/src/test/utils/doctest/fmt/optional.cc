#include "test/utils/doctest/fmt/optional.h"

namespace doctest {

String StringMaker<std::nullopt_t>::convert(std::nullopt_t const &m) {
  return toString(fmt::to_string(m));
}

} // namespace doctest
