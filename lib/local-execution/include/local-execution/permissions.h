#ifndef _FLEXFLOW_LOCAL_EXECUTION_PERMISSION_H
#define _FLEXFLOW_LOCAL_EXECUTION_PERMISSION_H

#include "utils/exception.h"
#include "utils/fmt.h"

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
        throw ::FlexFlow::mk_runtime_error(
            fmt::format("Unknown permission {}", static_cast<int>(p)));
    }
    return formatter<string_view>::format(name, ctx);
  }
};

} // namespace fmt

#endif
