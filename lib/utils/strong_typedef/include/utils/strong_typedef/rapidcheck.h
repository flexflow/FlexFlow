#ifndef _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_RAPIDCHECK_H
#define _FLEXFLOW_LIB_UTILS_STRONG_TYPEDEF_INCLUDE_UTILS_STRONG_TYPEDEF_RAPIDCHECK_H

#include "is_strong_typedef.h"
#include "rapidcheck/gen/Arbitrary.h"
#include "underlying_type.h"

namespace rc {

template <typename Tag>
struct Arbitrary<Tag, std::enable_if_t<is_strong_typedef<Tag>::value>> {
  static Gen<Tag> arbitrary() {
    return gen::construct<Tag>(
        gen::arbitrary<::FlexFlow::underlying_type_t<Tag>>());
  }
};

} // namespace rc

#endif
