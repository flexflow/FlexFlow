#include "local-execution/permissions.h"
#include "utils/exception.h"

namespace FlexFlow {

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
      throw mk_runtime_error(
          fmt::format("Unknown permission {}", static_cast<int>(p)));
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
