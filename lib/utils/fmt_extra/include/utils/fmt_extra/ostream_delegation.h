#ifndef _FLEXFLOW_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_OSTREAM_DELEGATION_H
#define _FLEXFLOW_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_OSTREAM_DELEGATION_H

namespace FlexFlow {

template <typename T>
struct ostream_operator_delegate_is_expected;

template <typename T>
inline constexpr bool ostream_operator_delegate_is_expected_v =
    ostream_operator_delegate_is_expected<T>::value;

}

#endif
