#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_FMT_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_FMT_DECL_H

#include "fmt/format.h"
#include "utils/test_types.h"
#include <list>
#include <map>
#include <set>
#include <tuple>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace FlexFlow {

template <typename T>
struct ostream_operator_delegate_is_expected;

template <typename T>
inline static constexpr bool ostream_operator_delegate_is_expected_v =
    ostream_operator_delegate_is_expected<T>::value;

template <typename T>
enable_if_t<is_fmtable_v<T> && !already_has_ostream_operator_v<T> &&
                !ostream_operator_delegate_is_expected_v<T>,
            std::ostream &>
    operator<<(std::ostream &s, T const &t);

template <typename T>
enable_if_t<ostream_operator_delegate_is_expected_v<T>, std::ostream &>
    operator<<(std::ostream &s, T const &t);

} // namespace FlexFlow

namespace fmt {

template <typename T>
struct formatter<::std::vector<T>,
                 ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char>>
    : formatter<::std::string> {
  auto format(::std::vector<T> const &m, format_context &ctx) const;
};

template <typename T>
struct formatter<::std::list<T>,
                 ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char>>
    : formatter<::std::string> {
  auto format(::std::list<T> const &m, format_context &ctx) const;
};

template <typename T>
struct formatter<::std::unordered_set<T>,
                 ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char>>
    : formatter<::std::string> {
  auto format(::std::unordered_set<T> const &m, format_context &ctx) const;
};

template <typename T>
struct formatter<::std::set<T>,
                 ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char>>
    : formatter<::std::string> {
  auto format(::std::set<T> const &m, format_context &ctx) const;
};

template <typename K, typename V>
struct formatter<::std::unordered_map<K, V>,
                 ::std::enable_if_t<::FlexFlow::all_is_fmtable_v<K, V>, char>>
    : formatter<::std::string> {
  auto format(::std::unordered_map<K, V> const &m, format_context &ctx) const;
};

template <typename K, typename V>
struct formatter<::std::map<K, V>,
                 ::std::enable_if_t<::FlexFlow::all_is_fmtable_v<K, V>, char>>
    : formatter<::std::string> {
  auto format(::std::map<K, V> const &m, format_context &ctx) const;
};

template <typename... Ts>
struct formatter<::std::tuple<Ts...>,
                 ::std::enable_if_t<::FlexFlow::all_is_fmtable_v<Ts...>, char>>
    : formatter<::std::string> {
  auto format(::std::tuple<Ts...> const &m, format_context &ctx) const;
};

template <typename T1, typename T2>
struct formatter<::std::pair<T1, T2>,
                 ::std::enable_if_t<::FlexFlow::all_is_fmtable_v<T1, T2>, char>>
    : formatter<::std::string> {
  auto format(::std::pair<T1, T2> const &m, format_context &ctx) const;
};

} // namespace fmt

namespace FlexFlow {
CHECK_FMTABLE(std::vector<test_types::fmtable>);
CHECK_FMTABLE(std::list<test_types::fmtable>);
CHECK_FMTABLE(std::unordered_set<test_types::wb_hash_fmt>);
CHECK_FMTABLE(std::set<test_types::wb_fmt>);
CHECK_FMTABLE(std::unordered_map<test_types::wb_hash_fmt, test_types::wb_fmt>);
CHECK_FMTABLE(std::map<test_types::wb_fmt, test_types::wb_fmt>);
CHECK_FMTABLE(std::tuple<test_types::fmtable,
                         test_types::fmtable,
                         test_types::fmtable,
                         test_types::fmtable>);
CHECK_FMTABLE(std::pair<test_types::fmtable, test_types::fmtable>);
} // namespace FlexFlow

#include "fmt.inl"

#endif
