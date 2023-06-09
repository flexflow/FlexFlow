#include "permissions.h"
#include "utils/exception.h"

namespace FlexFlow {

Legion::PrivilegeMode to_legion(Permissions p) {
  switch (p) {
    case Permissions::NONE:
      return LEGION_NO_ACCESS;
    case Permissions::RO:
      return LEGION_READ_ONLY;
    case Permissions::WO:
      return LEGION_WRITE_ONLY;
    case Permissions::RW:
      return LEGION_READ_WRITE;
    default:
      throw mk_runtime_error("Unknown permission {}", static_cast<int>(p));
  }
}

optional<Permissions> from_legion(Legion::PrivilegeMode p) {
  switch (p) {
    case LEGION_NO_ACCESS:
      return Permissions::NONE;
    case LEGION_READ_ONLY:
      return Permissions::RO;
    case LEGION_WRITE_ONLY:
      return Permissions::WO;
    case LEGION_READ_WRITE:
      return Permissions::RW;
    default:
      return nullopt;
  }
}

Permissions join(Permissions lhs, Permissions rhs) {
  if (lhs <= rhs) {
    return rhs;
  } else if (rhs <= lhs) {
    return lhs;
  } else {
    return Permissions::RW;
  }
}

Permissions meet(Permissions lhs, Permissions rhs) {
  if (lhs <= rhs) {
    return lhs;
  } else if (rhs <= lhs) {
    return rhs;
  } else {
    return Permissions::NONE;
  }
}

static int as_int(Permissions p) {
  switch (p) {
    case Permissions::NONE:
      return 0;
    case Permissions::RO:
    case Permissions::WO:
      return 1;
    case Permissions::RW:
      return 2;
    default:
      throw mk_runtime_error("Unknown permission {}", static_cast<int>(p));
  }
}

static bool comparable(Permissions lhs, Permissions rhs) {
  return !(lhs == Permissions::RO && rhs == Permissions::WO ||
           lhs == Permissions::WO && rhs == Permissions::RO);
}

bool operator<(Permissions lhs, Permissions rhs) {
  if (!comparable(lhs, rhs)) {
    return false;
  }
  int lhs_int = as_int(lhs);
  int rhs_int = as_int(rhs);
  return lhs_int < rhs_int;
}

bool operator<=(Permissions lhs, Permissions rhs) {
  return (lhs < rhs) || (lhs == rhs);
}

bool operator>(Permissions lhs, Permissions rhs) {
  if (!comparable(lhs, rhs)) {
    return false;
  }
  int lhs_int = as_int(lhs);
  int rhs_int = as_int(rhs);
  return lhs_int > rhs_int;
}

bool operator>=(Permissions lhs, Permissions rhs) {
  return (lhs > rhs) || (lhs == rhs);
}

} // namespace FlexFlow
