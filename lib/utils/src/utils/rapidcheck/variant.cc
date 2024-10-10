#include "utils/rapidcheck/variant.h"

namespace rc {

using T0 = int;
using T1 = std::string;

template struct Arbitrary<std::variant<T0, T1>>;

} // namespace rc
