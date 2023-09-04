#ifndef _FLEXFLOW_UTILS_INCLUDE_STRONG_TYPEDEF_H
#define _FLEXFLOW_UTILS_INCLUDE_STRONG_TYPEDEF_H

#include "utils/fmt.decl.h"
#include <functional>
#include <string>
#include <type_traits>

namespace FlexFlow {

// derived from https://www.foonathan.net/2016/10/strong-typedefs/
template <typename Tag, typename T>
class strong_typedef {
public:
  strong_typedef() = delete;

  explicit strong_typedef(T const &value);

  explicit strong_typedef(T &&value) noexcept(
      std::is_nothrow_move_constructible<T>::value);

  explicit operator T &() noexcept;

  explicit operator T const &() const noexcept;

  template <typename TT,
            typename = std::enable_if_t<(is_static_castable_v<T, TT> &&
                                         !std::is_same_v<T, TT>)>>
  explicit operator TT() const;

  template <typename TT,
            typename std::enable_if_t<(std::is_convertible_v<T, TT> &&
                                       !std::is_same_v<T, TT>),
                                      bool> = true>
  operator TT() const;

  friend void swap(strong_typedef &a, strong_typedef &b) noexcept;

  friend bool operator==(strong_typedef const &lhs, strong_typedef const &rhs);
  friend bool operator!=(strong_typedef const &lhs, strong_typedef const &rhs);
  friend bool operator<(strong_typedef const &lhs, strong_typedef const &rhs);

  T const &value() const noexcept;

  template <typename F>
  strong_typedef fmap(F const &f);

private:
  T value_;
};

template <typename Tag, typename T>
T underlying_type_impl(strong_typedef<Tag, T>);

template <typename T>
struct underlying_type
    : type_identity<decltype(underlying_type_impl(std::declval<T>()))> {};

template <typename T>
using underlying_type_t = typename underlying_type<T>::type;

template <typename T, typename Enable = void>
struct is_strong_typedef;
template <typename T>
inline constexpr bool is_strong_typedef_v = is_strong_typedef<T>::value;

// derived from
// https://github.com/foonathan/type_safe/blob/3612e2828b4b4e0d1cc689373e63a6d59d4bfd79/include/type_safe/strong_typedef.hpp
template <typename StrongTypedef>
struct hashable : std::hash<underlying_type_t<StrongTypedef>> {
  using underlying_ty = underlying_type_t<StrongTypedef>;
  using underlying_hash = std::hash<underlying_ty>;

  std::size_t operator()(StrongTypedef const &lhs) const
      noexcept(noexcept(underlying_hash{}(std::declval<underlying_ty>())));
};

template <typename StrongTypedef, typename T>
struct numerical_typedef : strong_typedef<StrongTypedef, T> {
  using strong_typedef<StrongTypedef, T>::strong_typedef;

  friend StrongTypedef &operator+=(StrongTypedef &lhs, T const &rhs);

  friend StrongTypedef &operator++(StrongTypedef &lhs);

  friend StrongTypedef operator++(StrongTypedef &lhs, int);

  friend StrongTypedef operator+(StrongTypedef const &lhs, T const &rhs);

  friend StrongTypedef operator+(T const &lhs, StrongTypedef const &rhs);
  friend StrongTypedef operator-=(StrongTypedef &lhs, T const &rhs);
  friend StrongTypedef &operator--(StrongTypedef &lhs);
  friend StrongTypedef operator--(StrongTypedef &lhs, int);
  friend StrongTypedef operator-(StrongTypedef const &lhs, T const &rhs);
  friend bool operator<(StrongTypedef const &lhs, T const &rhs);
  friend bool operator==(StrongTypedef const &lhs, T const &rhs);
  friend bool operator>(StrongTypedef const &lhs, T const &rhs);
  friend bool operator>=(StrongTypedef const &lhs, T const &rhs);
  friend bool operator!=(StrongTypedef const &lhs, T const &rhs);
  friend bool operator<=(StrongTypedef const &lhs, T const &rhs);
  friend bool operator<(T const &lhs, StrongTypedef const &rhs);
  friend bool operator==(T const &lhs, StrongTypedef const &rhs);
  friend bool operator>(T const &lhs, StrongTypedef const &rhs);
  friend bool operator<=(T const &lhs, StrongTypedef const &rhs);
  friend bool operator!=(T const &lhs, StrongTypedef const &rhs);
  friend bool operator>=(T const &lhs, StrongTypedef const &rhs);
};

} // namespace FlexFlow

#define MAKE_TYPEDEF_HASHABLE(TYPEDEF_NAME)                                    \
  namespace std {                                                              \
  template <>                                                                  \
  struct hash<TYPEDEF_NAME> : ::FlexFlow::hashable<TYPEDEF_NAME> {};           \
  }                                                                            \
  static_assert(true, "")

#define MAKE_TYPEDEF_PRINTABLE(TYPEDEF_NAME, TYPEDEF_SHORTNAME)                \
  namespace fmt {                                                              \
  template <>                                                                  \
  struct formatter<TYPEDEF_NAME> : formatter<::std::string> {                  \
    template <typename FormatContext>                                          \
    auto format(TYPEDEF_NAME const &x, FormatContext &ctx) const               \
        -> decltype(ctx.out()) {                                               \
      ::std::string s = fmt::format("{}({})", (TYPEDEF_SHORTNAME), x.value()); \
      return formatter<::std::string>::format(s, ctx);                         \
    }                                                                          \
  };                                                                           \
  }                                                                            \
  static_assert(true, "")

#define FF_TYPEDEF_HASHABLE(TYPEDEF_NAME)                                      \
  }                                                                            \
  MAKE_TYPEDEF_HASHABLE(::FlexFlow::TYPEDEF_NAME);                             \
  namespace FlexFlow {                                                         \
  static_assert(true, "");

#define FF_TYPEDEF_PRINTABLE(TYPEDEF_NAME, TYPEDEF_SHORTNAME)                  \
  }                                                                            \
  MAKE_TYPEDEF_PRINTABLE(::FlexFlow::TYPEDEF_NAME, TYPEDEF_SHORTNAME);         \
  namespace FlexFlow {                                                         \
  inline std::ostream &operator<<(std::ostream &s, TYPEDEF_NAME const &t) {    \
    return (s << fmt::to_string(t));                                           \
  }                                                                            \
  static_assert(true, "");

#endif
