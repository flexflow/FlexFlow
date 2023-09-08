#ifndef _FLEXFLOW_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_DO_NOT_DELEGATE_OSTREAM_OPERATOR_H
#define _FLEXFLOW_UTILS_FMT_EXTRA_INCLUDE_UTILS_FMT_EXTRA_DO_NOT_DELEGATE_OSTREAM_OPERATOR_H

namespace FlexFlow {

template <typename T>
struct do_not_delegate_ostream_operator;

template <typename T>
inline constexpr bool do_not_delegate_ostream_operator_v = do_not_delegate_ostream_operator<T>::value;

template <typename T>

template <> do_not_delegate_ostream_operator<int> : std::true_type {};

template <> do_not_delegate_ostream_operator<char> : std::true_type {};
template <size_t N> do_not_delegate_ostream_operator<char[N]> : std::true_type {};
template <> do_not_delegate_ostream_operator<char *> : std::true_type {};

}

#endif
