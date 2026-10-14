#ifndef _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_POSITIVE_INT_POSITIVE_INT_H
#define _FLEXFLOW_LIB_UTILS_INCLUDE_UTILS_POSITIVE_INT_POSITIVE_INT_H

#include "utils/nonnegative_int/nonnegative_int.h"

namespace FlexFlow {

struct positive_int {
  positive_int() = delete;
  explicit positive_int(int value);
  explicit positive_int(size_t value);
  explicit positive_int(nonnegative_int value);

  explicit operator int() const noexcept;
  explicit operator nonnegative_int() const noexcept;
  explicit operator float() const noexcept;

  bool operator<(positive_int other) const;
  bool operator==(positive_int other) const;
  bool operator>(positive_int other) const;
  bool operator<=(positive_int other) const;
  bool operator!=(positive_int other) const;
  bool operator>=(positive_int other) const;

  bool operator<(nonnegative_int other) const;
  bool operator==(nonnegative_int other) const;
  bool operator>(nonnegative_int other) const;
  bool operator<=(nonnegative_int other) const;
  bool operator!=(nonnegative_int other) const;
  bool operator>=(nonnegative_int other) const;

  friend bool operator<(nonnegative_int lhs, positive_int rhs);
  friend bool operator==(nonnegative_int lhs, positive_int rhs);
  friend bool operator>(nonnegative_int lhs, positive_int rhs);
  friend bool operator<=(nonnegative_int lhs, positive_int rhs);
  friend bool operator!=(nonnegative_int lhs, positive_int rhs);
  friend bool operator>=(nonnegative_int lhs, positive_int rhs);

  bool operator<(int other) const;
  bool operator==(int other) const;
  bool operator>(int other) const;
  bool operator<=(int other) const;
  bool operator!=(int other) const;
  bool operator>=(int other) const;

  friend bool operator<(int lhs, positive_int rhs);
  friend bool operator==(int lhs, positive_int rhs);
  friend bool operator>(int lhs, positive_int rhs);
  friend bool operator<=(int lhs, positive_int rhs);
  friend bool operator!=(int lhs, positive_int rhs);
  friend bool operator>=(int lhs, positive_int rhs);

  positive_int operator+(positive_int other) const;
  positive_int operator+(nonnegative_int other) const;
  positive_int &operator++();
  positive_int operator++(int);
  positive_int &operator+=(positive_int other);
  positive_int &operator+=(nonnegative_int other);

  friend positive_int operator+(nonnegative_int lhs, positive_int rhs);
  friend nonnegative_int &operator+=(nonnegative_int &lhs, positive_int rhs);

  positive_int operator*(positive_int other) const;
  positive_int &operator*=(positive_int other);
  nonnegative_int operator*(nonnegative_int other) const;

  friend nonnegative_int operator*(nonnegative_int lhs, positive_int rhs);

  float operator*(float other) const;
  friend float operator*(float lhs, positive_int rhs);

  int operator/(int other) const;
  nonnegative_int operator/(positive_int other) const;
  friend nonnegative_int operator/(nonnegative_int lhs, positive_int rhs);

  friend int &operator/=(int &lhs, positive_int rhs);

  friend float operator/(float lhs, positive_int rhs);
  friend float &operator/=(float &lhs, positive_int rhs);

  int operator%(int other) const;
  friend nonnegative_int operator%(int lhs, positive_int rhs);

  nonnegative_int operator%(positive_int other) const;
  friend nonnegative_int operator%(nonnegative_int lhs, positive_int rhs);

  int int_from_positive_int() const;
  size_t size_t_from_positive_int() const;
  nonnegative_int nonnegative_int_from_positive_int() const;

  friend std::ostream &operator<<(std::ostream &os, positive_int n);

  friend int format_as(positive_int);

private:
  void check_invariant() const;

private:
  int value_;
};

positive_int operator""_p(unsigned long long int);

} // namespace FlexFlow

namespace nlohmann {
template <>
struct adl_serializer<::FlexFlow::positive_int> {
  static ::FlexFlow::positive_int from_json(json const &j);
  static void to_json(json &j, ::FlexFlow::positive_int t);
};
} // namespace nlohmann

namespace rc {
template <>
struct Arbitrary<::FlexFlow::positive_int> {
  static Gen<::FlexFlow::positive_int> arbitrary();
};
} // namespace rc

namespace std {
template <>
struct hash<::FlexFlow::positive_int> {
  std::size_t operator()(FlexFlow::positive_int n) const noexcept;
};
} // namespace std

#endif
