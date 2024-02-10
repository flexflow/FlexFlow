#ifndef _FLEXFLOW_UTILS_INCLUDE_STRONG_TYPEDEF_H
#define _FLEXFLOW_UTILS_INCLUDE_STRONG_TYPEDEF_H

#include "utils/fmt.h"
#include <functional>
#include <string>
#include <type_traits>

namespace FlexFlow {

// derived from https://www.foonathan.net/2016/10/strong-typedefs/
template <typename Tag, typename T>
class strong_typedef {
public:
  strong_typedef() = delete;

  explicit strong_typedef(T const &value) : value_(value) {}

  explicit strong_typedef(T &&value) noexcept(
      std::is_nothrow_move_constructible<T>::value)
      : value_(std::move(value)) {}

  explicit operator T &() noexcept {
    return value_;
  }

  explicit operator T const &() const noexcept {
    return value_;
  }

  template <typename TT,
            typename std::enable_if<(is_static_castable<T, TT>::value &&
                                     !std::is_same<T, TT>::value),
                                    bool>::type = true>
  explicit operator TT() const {
    return static_cast<TT>(this->value_);
  }

  template <typename TT,
            typename std::enable_if<(std::is_convertible<T, TT>::value &&
                                     !std::is_same<T, TT>::value),
                                    bool>::type = true>
  operator TT() const {
    return (this->value_);
  }

  friend void swap(strong_typedef &a, strong_typedef &b) noexcept {
    using std::swap;
    swap(static_cast<T &>(a), static_cast<T &>(b));
  }

  friend bool operator==(strong_typedef const &lhs, strong_typedef const &rhs) {
    return lhs.value() == rhs.value();
  }

  friend bool operator!=(strong_typedef const &lhs, strong_typedef const &rhs) {
    return lhs.value() != rhs.value();
  }

  friend bool operator<(strong_typedef const &lhs, strong_typedef const &rhs) {
    return lhs.value() < rhs.value();
  }

  T const &value() const noexcept {
    return value_;
  }

  template <typename F>
  strong_typedef fmap(F const &f) {
    static_assert(
        std::is_same<decltype(std::declval<F>()(std::declval<T const &>())),
                     T>::value,
        "Function must return an value of the underlying type");

    return strong_typedef(f(this->value_));
  }

private:
  T value_;
};

template <typename Tag, typename T>
T underlying_type_impl(strong_typedef<Tag, T>);

template <typename T>
using underlying_type_t = decltype(underlying_type_impl(std::declval<T>()));
// derived from
// https://github.com/foonathan/type_safe/blob/3612e2828b4b4e0d1cc689373e63a6d59d4bfd79/include/type_safe/strong_typedef.hpp
template <typename StrongTypedef>
struct hashable : std::hash<underlying_type_t<StrongTypedef>> {
  using underlying_ty = underlying_type_t<StrongTypedef>;
  using underlying_hash = std::hash<underlying_ty>;

  std::size_t operator()(StrongTypedef const &lhs) const
      noexcept(noexcept(underlying_hash{}(std::declval<underlying_ty>()))) {
    return underlying_hash{}(static_cast<underlying_ty const &>(lhs));
  }
};

template <typename StrongTypedef, typename T>
struct numerical_typedef : strong_typedef<StrongTypedef, T> {
  using strong_typedef<StrongTypedef, T>::strong_typedef;

  friend StrongTypedef &operator+=(StrongTypedef &lhs, T const &rhs) {
    static_cast<T &>(lhs) += static_cast<T const &>(rhs);
    return lhs;
  }

  friend StrongTypedef &operator++(StrongTypedef &lhs) {
    static_cast<T &>(lhs) += static_cast<T>(1);
    return lhs;
  }

  friend StrongTypedef operator++(StrongTypedef &lhs, int) {
    StrongTypedef tmp = lhs;
    ++lhs;
    return tmp;
  }

  friend StrongTypedef operator+(StrongTypedef const &lhs, T const &rhs) {
    return StrongTypedef(lhs.value() + rhs);
  }

  friend StrongTypedef operator+(T const &lhs, StrongTypedef const &rhs) {
    return (rhs + lhs);
  }

  friend StrongTypedef operator-=(StrongTypedef &lhs, T const &rhs) {
    static_cast<T &>(lhs) -= static_cast<T const &>(rhs);
  }

  friend StrongTypedef &operator--(StrongTypedef &lhs) {
    static_cast<T &>(lhs) -= static_cast<T>(1);
    return lhs;
  }

  friend StrongTypedef operator--(StrongTypedef &lhs, int) {
    StrongTypedef tmp = lhs;
    --lhs;
    return tmp;
  }

  friend StrongTypedef operator-(StrongTypedef const &lhs, T const &rhs) {
    return StrongTypedef(lhs.value() + rhs);
  }

  friend bool operator<(StrongTypedef const &lhs, T const &rhs) {
    return lhs.value() < rhs;
  }

  friend bool operator==(StrongTypedef const &lhs, T const &rhs) {
    return lhs.value() == rhs;
  }

  friend bool operator>(StrongTypedef const &lhs, T const &rhs) {
    return lhs.value() > rhs;
  }

  friend bool operator>=(StrongTypedef const &lhs, T const &rhs) {
    return lhs.value() >= rhs;
  }

  friend bool operator!=(StrongTypedef const &lhs, T const &rhs) {
    return lhs.value() != rhs;
  }

  friend bool operator<=(StrongTypedef const &lhs, T const &rhs) {
    return lhs.value() <= rhs;
  }

  friend bool operator<(T const &lhs, StrongTypedef const &rhs) {
    return lhs < rhs.value();
  }

  friend bool operator==(T const &lhs, StrongTypedef const &rhs) {
    return lhs == rhs.value();
  }

  friend bool operator>(T const &lhs, StrongTypedef const &rhs) {
    return lhs > rhs.value();
  }

  friend bool operator<=(T const &lhs, StrongTypedef const &rhs) {
    return lhs <= rhs.value();
  }

  friend bool operator!=(T const &lhs, StrongTypedef const &rhs) {
    return lhs != rhs.value();
  }

  friend bool operator>=(T const &lhs, StrongTypedef const &rhs) {
    return lhs >= rhs.value();
  }
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
  static_assert(true, "");

#endif
