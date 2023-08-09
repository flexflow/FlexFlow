#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_OPS_CORE_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_OPS_CORE_H

#include "utils/type_traits.h"

namespace FlexFlow {

#define CHECK_VALID_OP_ATTR(TYPENAME) CHECK_WELL_BEHAVED_VALUE_TYPE(TYPENAME)

template <typename T, typename Enable = void>
using is_valid_opattr = is_well_behaved_value_type<T>;

} // namespace FlexFlow

#endif
