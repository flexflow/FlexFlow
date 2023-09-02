#ifndef _FLEXFLOW_UTILS_INCLUDE_FMT_H
#define _FLEXFLOW_UTILS_INCLUDE_FMT_H

#include "utils/containers.decl.h"
#include "utils/fmt.decl.h"
#include "utils/test_types.h"
#include "utils/type_traits_core.h"
#include "utils/exception.decl.h"
#include "utils/string.h"

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

template <typename T>
enable_if_t<!already_has_ostream_operator_v<T>, std::ostream &>
    operator<<(std::ostream &s, T const &t) {
  CHECK_FMTABLE(T);

  std::string result = fmt::to_string(t);
  return s << result;
}

template <typename T>
std::string element_to_string(T const &t) {
  return fmt::to_string(t);
}

std::string element_to_string(char const s[]);
template <> std::string element_to_string(std::string const &s);
template <> std::string element_to_string(char const &c);

}

namespace fmt {

template <typename T>
auto formatter<
  ::std::vector<T>, 
  ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char>
>::format(::std::vector<T> const &m, format_context &ctx) const {
  using namespace ::FlexFlow;

  std::string result = surrounded('[', ']', ::FlexFlow::join_strings(m, ", ", [](T const &t) { return element_to_string(t); }));
  return formatter<std::string>::format(result, ctx);
}

template <typename T>
auto formatter<
  ::std::list<T>, 
  ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char>
>::format(::std::list<T> const &m, format_context &ctx) const {
  using namespace ::FlexFlow;

  std::string result = surrounded('[', ']', join_strings(m, ", ", [](T const &t) { return element_to_string(t); }));
  return formatter<std::string>::format(result, ctx);
}

template <typename T>
auto formatter<
  ::std::unordered_set<T>, 
  ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char>
>::format(::std::unordered_set<T> const &m, format_context &ctx) const {
  using namespace ::FlexFlow;

  std::string result = surrounded('{', '}', join_strings(sorted(m), ", ", [](T const &t) { return element_to_string(t); }));
  return formatter<std::string>::format(result, ctx);
}

template <typename T>
auto formatter<
  ::std::set<T>, 
  ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char>
>::format(::std::set<T> const &m, format_context &ctx) const {
  using namespace ::FlexFlow;

  std::string result = surrounded('{', '}', join_strings(sorted(m), ", ", [](T const &t) { return element_to_string(t); }));
  return formatter<std::string>::format(result, ctx);
}

template <typename K, typename V>
auto formatter<
  ::std::unordered_map<K, V>, 
  ::std::enable_if_t<::FlexFlow::all_is_fmtable_v<K, V>, char> 
>::format(::std::unordered_map<K, V> const &m, format_context &ctx) const {
  using namespace ::FlexFlow;

  std::string result = surrounded('{', '}', join_strings(sorted(m), ", ", [](std::pair<K, V> const &kv) { return fmt::format("{}: {}", element_to_string(kv.first), element_to_string(kv.second)); }));
  return formatter<std::string>::format(result, ctx);
}

template <typename K, typename V>
auto formatter<
  ::std::map<K, V>, 
  ::std::enable_if_t<::FlexFlow::all_is_fmtable_v<K, V>, char> 
>::format(::std::map<K, V> const &m, format_context &ctx) const {
  using namespace ::FlexFlow;

  std::string result = surrounded('{', '}', join_strings(sorted(m), ", ", [](std::pair<K, V> const &kv) { return fmt::format("{}: {}", element_to_string(kv.first), element_to_string(kv.second)); }));
  return formatter<std::string>::format(result, ctx);
}

template <typename T1, typename T2>
auto formatter<
  ::std::pair<T1, T2>, 
  ::std::enable_if_t<::FlexFlow::all_is_fmtable_v<T1, T2>, char> 
>::format(::std::pair<T1, T2> const &m, format_context &ctx) const {
  using namespace ::FlexFlow;

  std::string result = fmt::format("<{}, {}>", element_to_string(m.first), element_to_string(m.second));
  return formatter<std::string>::format(result, ctx);
}

template <typename... Ts>
auto formatter<
  ::std::tuple<Ts...>, 
  ::std::enable_if_t<::FlexFlow::all_is_fmtable_v<Ts...>, char> 
>::format(::std::tuple<Ts...> const &m, format_context &ctx) const {
  using namespace ::FlexFlow;

  std::vector<std::string> v = to_vector(transform(m, [](auto const &t) { return element_to_string(t); }));
  std::string result = surrounded('<', '>', join_strings(v, ", ", [](std::string const &s) { return s; }));
  return formatter<std::string>::format(result, ctx);
}

} // namespace fmt

#endif
