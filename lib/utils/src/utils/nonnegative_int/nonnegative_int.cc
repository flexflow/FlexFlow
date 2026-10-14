#include "utils/nonnegative_int/nonnegative_int.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

nonnegative_int::nonnegative_int(int value) {
  ASSERT(value >= 0, "Value of nonnegative_int must be nonnegative");
  this->value_ = value;
}

nonnegative_int::nonnegative_int(size_t value) {
  ASSERT(value <= std::numeric_limits<int>::max());
  this->value_ = static_cast<int>(value);
  ASSERT(this->value_ >= 0, "Value of nonnegative_int must be nonnegative");
}

nonnegative_int::nonnegative_int(unsigned long long int value) {
  ASSERT(value <= std::numeric_limits<int>::max());
  this->value_ = static_cast<int>(value);
  ASSERT(this->value_ >= 0, "Value of nonnegative_int must be nonnegative");
}

nonnegative_int::operator int() const noexcept {
  return this->value_;
}

nonnegative_int::operator size_t() const noexcept {
  return static_cast<size_t>(this->value_);
}

bool nonnegative_int::operator<(nonnegative_int const &other) const {
  return this->value_ < other.value_;
}

bool nonnegative_int::operator==(nonnegative_int const &other) const {
  return this->value_ == other.value_;
}

bool nonnegative_int::operator>(nonnegative_int const &other) const {
  return this->value_ > other.value_;
}

bool nonnegative_int::operator<=(nonnegative_int const &other) const {
  return this->value_ <= other.value_;
}

bool nonnegative_int::operator!=(nonnegative_int const &other) const {
  return this->value_ != other.value_;
}

bool nonnegative_int::operator>=(nonnegative_int const &other) const {
  return this->value_ >= other.value_;
}

bool nonnegative_int::operator<(int const &other) const {
  return this->value_ < other;
}

bool nonnegative_int::operator==(int const &other) const {
  return this->value_ == other;
}

bool nonnegative_int::operator>(int const &other) const {
  return this->value_ > other;
}

bool nonnegative_int::operator<=(int const &other) const {
  return this->value_ <= other;
}

bool nonnegative_int::operator!=(int const &other) const {
  return this->value_ != other;
}

bool nonnegative_int::operator>=(int const &other) const {
  return this->value_ >= other;
}

bool operator<(int const &lhs, nonnegative_int const &rhs) {
  return lhs < rhs.value_;
}

bool operator==(int const &lhs, nonnegative_int const &rhs) {
  return lhs == rhs.value_;
}

bool operator>(int const &lhs, nonnegative_int const &rhs) {
  return lhs > rhs.value_;
}

bool operator<=(int const &lhs, nonnegative_int const &rhs) {
  return lhs <= rhs.value_;
}

bool operator!=(int const &lhs, nonnegative_int const &rhs) {
  return lhs != rhs.value_;
}

bool operator>=(int const &lhs, nonnegative_int const &rhs) {
  return lhs >= rhs.value_;
}

nonnegative_int nonnegative_int::operator+(nonnegative_int const &other) const {
  return nonnegative_int{this->value_ + other.value_};
}

nonnegative_int &nonnegative_int::operator++() {
  this->value_++;
  return *this;
}

nonnegative_int nonnegative_int::operator++(int) {
  nonnegative_int result = *this;
  this->value_++;
  return result;
}

nonnegative_int &nonnegative_int::operator+=(nonnegative_int const &other) {
  this->value_ += other.value_;
  return *this;
}

nonnegative_int nonnegative_int::operator*(nonnegative_int const &other) const {
  return nonnegative_int{this->value_ * other.value_};
}

nonnegative_int &nonnegative_int::operator*=(nonnegative_int const &other) {
  this->value_ *= other.value_;
  return *this;
}

float nonnegative_int::operator*(float other) const {
  return this->value_ * other;
}

float operator*(float lhs, nonnegative_int rhs) {
  return lhs * rhs.value_;
}

float &operator*=(float &lhs, nonnegative_int rhs) {
  return (lhs *= rhs.value_);
}

int nonnegative_int::operator/(int other) const {
  return (this->value_ / other);
}

nonnegative_int nonnegative_int::operator/(nonnegative_int const &other) const {
  return nonnegative_int{this->value_ / other.value_};
}

nonnegative_int &nonnegative_int::operator/=(nonnegative_int const &other) {
  this->value_ /= other.value_;
  return *this;
}

int operator/(int lhs, nonnegative_int rhs) {
  return (lhs / rhs.value_);
}

int &operator/=(int &lhs, nonnegative_int rhs) {
  return (lhs /= rhs.value_);
}

float operator/(float lhs, nonnegative_int rhs) {
  return lhs / rhs.value_;
}

float &operator/=(float &lhs, nonnegative_int rhs) {
  return (lhs /= rhs.value_);
}

nonnegative_int nonnegative_int::operator%(nonnegative_int const &other) const {
  return nonnegative_int{this->value_ % other.value_};
}

nonnegative_int &nonnegative_int::operator%=(nonnegative_int const &other) {
  this->value_ %= other.value_;
  return *this;
}

int nonnegative_int::int_from_nonnegative_int() const {
  return this->value_;
}

size_t nonnegative_int::size_t_from_nonnegative_int() const {
  return static_cast<size_t>(this->value_);
}

std::ostream &operator<<(std::ostream &os, nonnegative_int const &n) {
  os << n.value_;
  return os;
}

int nonnegative_int::unwrap_nonnegative() const {
  return this->value_;
}

int format_as(nonnegative_int const &x) {
  return x.unwrap_nonnegative();
}

nonnegative_int operator""_n(unsigned long long int x) {
  if (x >
      static_cast<unsigned long long int>(std::numeric_limits<int>::max())) {
    PANIC(fmt::format("Value too large to wrap as nonnegative_int: {}", x));
  }

  return nonnegative_int{static_cast<int>(x)};
}

} // namespace FlexFlow

namespace nlohmann {
::FlexFlow::nonnegative_int
    adl_serializer<::FlexFlow::nonnegative_int>::from_json(json const &j) {
  return ::FlexFlow::nonnegative_int{j.template get<int>()};
}

void adl_serializer<::FlexFlow::nonnegative_int>::to_json(
    json &j, ::FlexFlow::nonnegative_int t) {
  j = t.unwrap_nonnegative();
}
} // namespace nlohmann

namespace rc {
Gen<::FlexFlow::nonnegative_int>
    Arbitrary<::FlexFlow::nonnegative_int>::arbitrary() {
  return gen::construct<::FlexFlow::nonnegative_int>(gen::nonNegative<int>());
}
} // namespace rc

namespace std {
std::size_t hash<::FlexFlow::nonnegative_int>::operator()(
    FlexFlow::nonnegative_int const &n) const noexcept {
  return std::hash<int>{}(n.unwrap_nonnegative());
}
} // namespace std
