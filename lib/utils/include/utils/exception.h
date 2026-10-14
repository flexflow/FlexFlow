#ifndef _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H
#define _FLEXFLOW_UTILS_INCLUDE_EXCEPTION_H

#include "utils/fmt.h"
#include <fmt/format.h>
#include <libassert/assert.hpp>
#include <stdexcept>
#include <tl/expected.hpp>

namespace FlexFlow {

template <typename T, typename E>
T throw_if_unexpected(tl::expected<T, E> const &r) {
  if (r.has_value()) {
    return r.value();
  } else {
    PANIC(fmt::to_string(r.error()));
    ;
  }
}

std::runtime_error mk_runtime_error(std::string const &);

} // namespace FlexFlow

#endif
