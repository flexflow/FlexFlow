#ifndef _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_PERMISSIONS_H
#define _FLEXFLOW_LIB_TASK_SPEC_INCLUDE_TASK_SPEC_PERMISSIONS_H

#include "utils/fmt.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

enum class Permissions { NONE, RO, WO, RW };

Permissions join(Permissions lhs, Permissions rhs);
Permissions meet(Permissions lhs, Permissions rhs);

bool operator<(Permissions lhs, Permissions rhs);
bool operator<=(Permissions lhs, Permissions rhs);
bool operator>(Permissions lhs, Permissions rhs);
bool operator>=(Permissions lhs, Permissions rhs);

} // namespace FlexFlow

namespace fmt {

template <>
struct formatter<::FlexFlow::Permissions> : formatter<string_view> {
  template <typename FormatContext>
  auto format(::FlexFlow::Permissions p, FormatContext &ctx) const
      -> decltype(ctx.out()) {
    using ::FlexFlow::Permissions;

    string_view name = "unknown";
    switch (p) {
      case Permissions::NONE:
        name = "NO_PERMISSIONS";
        break;
      case Permissions::RO:
        name = "READ_ONLY";
        break;
      case Permissions::WO:
        name = "WRITE_ONLY";
        break;
      case Permissions::RW:
        name = "READ_WRITE";
        break;
      default:
        PANIC(fmt::format("Unknown permission {}", static_cast<int>(p)));
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt

#endif
