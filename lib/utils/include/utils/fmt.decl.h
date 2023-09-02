#ifndef _FLEXFLOW_UTILS_INCLUDE_UTILS_FMT_DECL_H
#define _FLEXFLOW_UTILS_INCLUDE_UTILS_FMT_DECL_H

#include "fmt/format.h"
#include <unordered_set>
#include <vector>
#include <set>
#include <unordered_map>
#include <map>
#include <tuple>
#include <utility>
#include <list>
#include "utils/test_types.h"

#define CHECK_FMTABLE(...)                                                     \
  static_assert(::FlexFlow::is_fmtable<__VA_ARGS__>::value,                    \
                #__VA_ARGS__ " must be fmtable");

namespace FlexFlow {

template <typename T>
using is_fmtable = ::fmt::is_formattable<T>;

template <typename... Ts>
using all_is_fmtable = std::conjunction<is_fmtable<Ts>...>;

template <typename... Ts>
inline constexpr bool all_is_fmtable_v = all_is_fmtable<Ts...>::value;

template <typename T>
inline constexpr bool is_fmtable_v = is_fmtable<T>::value;

template <typename T, typename Enable = void>
struct already_has_ostream_operator;

template <typename T, typename Enable = void>
inline constexpr int already_has_ostream_operator_v = already_has_ostream_operator<T>::value;

template <typename T>
enable_if_t<is_fmtable_v<T> && !already_has_ostream_operator_v<T>, std::ostream &>
    operator<<(std::ostream &s, T const &t);

} // namespace FlexFlow

namespace fmt {

template <typename T>
struct formatter<
  ::std::vector<T>,
  ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char> 
> : formatter<::std::string>
{
  auto format(::std::vector<T> const &m, format_context &ctx) const;
};

template <typename T>
struct formatter<
  ::std::list<T>,
  ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char> 
> : formatter<::std::string>
{
  auto format(::std::list<T> const &m, format_context &ctx) const;
};

template <typename T>
struct formatter<
  ::std::unordered_set<T>,
  ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char> 
> : formatter<::std::string>
{
  auto format(::std::unordered_set<T> const &m, format_context &ctx) const;
};

template <typename T>
struct formatter<
  ::std::set<T>,
  ::std::enable_if_t<::FlexFlow::is_fmtable_v<T>, char> 
> : formatter<::std::string>
{
  auto format(::std::set<T> const &m, format_context &ctx) const;
};

template <typename K, typename V>
struct formatter<
  ::std::unordered_map<K, V>,
  ::std::enable_if_t<::FlexFlow::all_is_fmtable_v<K, V>, char> 
> : formatter<::std::string>
{
  auto format(::std::unordered_map<K, V> const &m, format_context &ctx) const;
};

template <typename K, typename V>
struct formatter<
  ::std::map<K, V>,
  ::std::enable_if_t<::FlexFlow::all_is_fmtable_v<K, V>, char> 
> : formatter<::std::string>
{
  auto format(::std::map<K, V> const &m, format_context &ctx) const;
};

template <typename... Ts>
struct formatter<
  ::std::tuple<Ts...>,
  ::std::enable_if_t<::FlexFlow::all_is_fmtable_v<Ts...>, char> 
> : formatter<::std::string>
{
  auto format(::std::tuple<Ts...> const &m, format_context &ctx) const;
};

template <typename T1, typename T2>
struct formatter<
  ::std::pair<T1, T2>,
  ::std::enable_if_t<::FlexFlow::all_is_fmtable_v<T1, T2>, char> 
> : formatter<::std::string>
{
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
  CHECK_FMTABLE(std::tuple<
                  test_types::fmtable, 
                  test_types::fmtable,
                  test_types::fmtable,
                  test_types::fmtable
                  >);
  CHECK_FMTABLE(std::pair<test_types::fmtable, test_types::fmtable>);
}

#endif
