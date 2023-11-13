#ifndef _FLEXFLOW_UTILS_INCLUDE_FMT_H
#define _FLEXFLOW_UTILS_INCLUDE_FMT_H

#include "utils/containers.decl.h"
#include "utils/fmt.decl.h"
#include "utils/test_types.h"
#include "utils/type_traits_core.h"

namespace FlexFlow {

template <typename T, typename Enable>
struct already_has_ostream_operator : std::false_type {};

template <>
struct already_has_ostream_operator<int> : std::true_type {};

template <>
struct already_has_ostream_operator<char> : std::true_type {};

template <>
struct already_has_ostream_operator<std::string> : std::true_type {};

template <size_t N>
struct already_has_ostream_operator<char[N]> : std::true_type {};

template <>
struct already_has_ostream_operator<char const *> : std::true_type {};

// This will create an error
/*
template <typename T>
std::ostream &
operator<<(std::ostream &s, T const &t) {
  return s << "FlexFlow::ostream<<";
}
*/

#define CHECK_FMTABLE(...)                                                     \
  static_assert(::FlexFlow::is_fmtable<__VA_ARGS__>::value,                    \
                #__VA_ARGS__ " must be fmtable");

// This will not
template <typename T>
typename std::enable_if<!already_has_ostream_operator<T>::value,
                        std::ostream &>::type
    operator<<(std::ostream &s, T const &t) {
  // CHECK_FMTABLE(T);

  std::string result = fmt::to_string(t);
  return s << result;
}

// template <typename T>
// typename std::enable_if<is_fmtable<T>::value, std::ostream &>::type
// operator<<(std::ostream &s, T const &t) {
//     return s << fmt::to_string(t);
// }

} // namespace FlexFlow

namespace fmt {

template <typename T>
template <typename FormatContext>
auto formatter<::std::unordered_set<T>>::format(
    ::std::unordered_set<T> const &m, FormatContext &ctx)
    -> decltype(ctx.out()) {
  // CHECK_FMTABLE(T);

  std::string result = join_strings(
      m.cbegin(), m.cend(), ", ", [](T const &t) { return fmt::to_string(t); });
  return formatter<std::string>::format(result, ctx);
}

template <typename T>
template <typename FormatContext>
auto formatter<::std::vector<T>>::format(::std::vector<T> const &m,
                                         FormatContext &ctx)
    -> decltype(ctx.out()) {
  // CHECK_FMTABLE(T);
  std::string result = join_strings(
      m.cbegin(), m.cend(), ", ", [](T const &t) { return fmt::to_string(t); });
  return formatter<std::string>::format(result, ctx);
}

/*template <typename Key, typename T,typename Hash, typename KeyEqual, typename
Allocator> template <typename FormatContext> auto
formatter<::std::unordered_map<Key, T,Hash, KeyEqual, Allocator>>::format(
    ::std::unordered_map<Key, T, Hash,  KeyEqual, Allocator> const & m,
FormatContext& ctx) -> decltype(ctx.out()) { std::string result = "1";
    join_strings(
        m.begin(), m.end(), ", ",
        [](const typename std::unordered_map<Key, T, Hash, KeyEqual,
Allocator>::value_type& entry) {
            // Format each entry as "key: value"
            return fmt::to_string(entry.first);
        });

    return formatter<std::string>::format(result, ctx);
}

/*template <typename T, typename U>
template <typename FormatContext>
auto formatter<::std::pair<T, U>>::format(const std::pair<T, U>& p,
FormatContext& ctx) -> decltype(ctx.out()) { return
formatter<std::string>::format(fmt::to_string(p.first), ctx);
}*/

// CHECK_FMTABLE(std::vector<int>);
// CHECK_FMTABLE(std::unordered_set<int>);

} // namespace fmt

#endif
