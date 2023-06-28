#ifndef _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_OPS_CORE_H
#define _FLEXFLOW_OPATTRS_INCLUDE_OPATTRS_OPS_CORE_H

#include "utils/type_traits.h"

namespace FlexFlow {

#define CHECK_VALID_OP_ATTR(TYPENAME) \
  CHECK_WELL_BEHAVED_VALUE_TYPE(TYPENAME)
  

template <typename T, typename Enable = void>
using is_valid_opattr = conjunction<is_equal_comparable<T>,
                                    is_neq_comparable<T>,
                                    is_lt_comparable<T>,
                                    is_hashable<T>,
                                    is_copy_constructible<T>,
                                    is_move_constructible<T>,
                                    is_copy_assignable<T>,
                                    is_move_assignable<T>>;

}

#endif
