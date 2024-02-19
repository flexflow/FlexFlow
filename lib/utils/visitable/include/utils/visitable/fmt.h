#ifndef _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_FMT_H
#define _FLEXFLOW_LIB_UTILS_VISITABLE_INCLUDE_UTILS_VISITABLE_FMT_H

namespace FlexFlow {

template <typename T>
struct ostream_operator_delegate_is_expected
    : disjunction<is_visitable<T>, is_strong_typedef<T>> {};

} // namespace FlexFlow

#endif
