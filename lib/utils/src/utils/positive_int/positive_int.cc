#include "utils/positive_int/positive_int.h"
#include "utils/integer_conversions.h"
#include <libassert/assert.hpp>

namespace FlexFlow {

positive_int::positive_int(int value) : value_(value) {
  this->check_invariant();
}

positive_int::positive_int(size_t value) : value_(value) {
  this->check_invariant();
}

positive_int::positive_int(nonnegative_int value)
    : value_(value.unwrap_nonnegative()) {
  this->check_invariant();
}

positive_int::operator int() const noexcept {
  return this->value_;
}

positive_int::operator nonnegative_int() const noexcept {
  return nonnegative_int{this->value_};
}

positive_int::operator float() const noexcept {
  return static_cast<float>(this->value_);
}

bool positive_int::operator<(positive_int other) const {
  return this->value_ < other.value_;
}

bool positive_int::operator==(positive_int other) const {
  return this->value_ == other.value_;
}

bool positive_int::operator>(positive_int other) const {
  return this->value_ > other.value_;
}

bool positive_int::operator<=(positive_int other) const {
  return this->value_ <= other.value_;
}

bool positive_int::operator!=(positive_int other) const {
  return this->value_ != other.value_;
}

bool positive_int::operator>=(positive_int other) const {
  return this->value_ >= other.value_;
}

bool positive_int::operator<(nonnegative_int other) const {
  return this->value_ < other;
}

bool positive_int::operator==(nonnegative_int other) const {
  return this->value_ == other;
}

bool positive_int::operator>(nonnegative_int other) const {
  return this->value_ > other;
}

bool positive_int::operator<=(nonnegative_int other) const {
  return this->value_ <= other;
}

bool positive_int::operator!=(nonnegative_int other) const {
  return this->value_ != other;
}

bool positive_int::operator>=(nonnegative_int other) const {
  return this->value_ >= other;
}

bool operator<(nonnegative_int lhs, positive_int rhs) {
  return lhs < rhs.value_;
}

bool operator==(nonnegative_int lhs, positive_int rhs) {
  return lhs == rhs.value_;
}

bool operator>(nonnegative_int lhs, positive_int rhs) {
  return lhs > rhs.value_;
}

bool operator<=(nonnegative_int lhs, positive_int rhs) {
  return lhs <= rhs.value_;
}

bool operator!=(nonnegative_int lhs, positive_int rhs) {
  return lhs != rhs.value_;
}

bool operator>=(nonnegative_int lhs, positive_int rhs) {
  return lhs >= rhs.value_;
}

bool positive_int::operator<(int other) const {
  return this->value_ < other;
}

bool positive_int::operator==(int other) const {
  return this->value_ == other;
}

bool positive_int::operator>(int other) const {
  return this->value_ > other;
}

bool positive_int::operator<=(int other) const {
  return this->value_ <= other;
}

bool positive_int::operator!=(int other) const {
  return this->value_ != other;
}

bool positive_int::operator>=(int other) const {
  return this->value_ >= other;
}

bool operator<(int lhs, positive_int rhs) {
  return lhs < rhs.value_;
}

bool operator==(int lhs, positive_int rhs) {
  return lhs == rhs.value_;
}

bool operator>(int lhs, positive_int rhs) {
  return lhs > rhs.value_;
}

bool operator<=(int lhs, positive_int rhs) {
  return lhs <= rhs.value_;
}

bool operator!=(int lhs, positive_int rhs) {
  return lhs != rhs.value_;
}

bool operator>=(int lhs, positive_int rhs) {
  return lhs >= rhs.value_;
}

positive_int positive_int::operator+(positive_int other) const {
  return positive_int{this->value_ + other.value_};
}

positive_int positive_int::operator+(nonnegative_int other) const {
  return positive_int{this->value_ + other.unwrap_nonnegative()};
}

nonnegative_int &operator+=(nonnegative_int &lhs, positive_int rhs) {
  return (lhs += rhs.nonnegative_int_from_positive_int());
}

positive_int &positive_int::operator++() {
  this->value_++;
  this->check_invariant();
  return *this;
}

positive_int positive_int::operator++(int) {
  positive_int result = *this;
  this->value_++;
  this->check_invariant();
  return result;
}

positive_int &positive_int::operator+=(positive_int other) {
  this->value_ += other.value_;
  this->check_invariant();
  return *this;
}

positive_int &positive_int::operator+=(nonnegative_int other) {
  this->value_ += other.unwrap_nonnegative();
  this->check_invariant();
  return *this;
}

positive_int operator+(nonnegative_int lhs, positive_int rhs) {
  return rhs + lhs;
}

positive_int positive_int::operator*(positive_int other) const {
  return positive_int{this->value_ * other.value_};
}

positive_int &positive_int::operator*=(positive_int other) {
  this->value_ *= other.value_;
  this->check_invariant();
  return *this;
}

nonnegative_int positive_int::operator*(nonnegative_int other) const {
  return other * *this;
}

nonnegative_int operator*(nonnegative_int lhs, positive_int rhs) {
  return lhs * rhs.nonnegative_int_from_positive_int();
}

float positive_int::operator*(float other) const {
  return this->value_ * other;
}

float operator*(float lhs, positive_int rhs) {
  return lhs * rhs.value_;
}

int positive_int::operator/(int other) const {
  return (this->value_ / other);
}

nonnegative_int positive_int::operator/(positive_int other) const {
  return nonnegative_int{this->value_ / other.value_};
}

nonnegative_int operator/(nonnegative_int lhs, positive_int rhs) {
  return nonnegative_int{lhs.unwrap_nonnegative() / rhs.value_};
}

int &operator/=(int &lhs, positive_int rhs) {
  return (lhs /= rhs.value_);
}

float operator/(float lhs, positive_int rhs) {
  return lhs / rhs.value_;
}

float &operator/=(float &lhs, positive_int rhs) {
  return (lhs /= rhs.value_);
}

nonnegative_int positive_int::operator%(positive_int other) const {
  return nonnegative_int{this->value_ % other.value_};
}

nonnegative_int operator%(nonnegative_int lhs, positive_int rhs) {
  return nonnegative_int{lhs.unwrap_nonnegative() % rhs.value_};
}

int positive_int::int_from_positive_int() const {
  return this->value_;
}

size_t positive_int::size_t_from_positive_int() const {
  return size_t_from_int(this->value_);
}

nonnegative_int positive_int::nonnegative_int_from_positive_int() const {
  return nonnegative_int{this->value_};
}

std::ostream &operator<<(std::ostream &os, positive_int n) {
  os << n.value_;
  return os;
}

int format_as(positive_int x) {
  return x.value_;
}

void positive_int::check_invariant() const {
  ASSERT(this->value_ > 0);
}

positive_int operator""_p(unsigned long long int x) {
  ASSERT(x <=
         static_cast<unsigned long long int>(std::numeric_limits<int>::max()));

  return positive_int{static_cast<int>(x)};
}

} // namespace FlexFlow

namespace nlohmann {
::FlexFlow::positive_int
    adl_serializer<::FlexFlow::positive_int>::from_json(json const &j) {
  return ::FlexFlow::positive_int{j.template get<int>()};
}

void adl_serializer<::FlexFlow::positive_int>::to_json(
    json &j, ::FlexFlow::positive_int t) {
  j = t.int_from_positive_int();
}
} // namespace nlohmann

namespace rc {
Gen<::FlexFlow::positive_int> Arbitrary<::FlexFlow::positive_int>::arbitrary() {
  return gen::construct<::FlexFlow::positive_int>(gen::positive<int>());
}
} // namespace rc

namespace std {
std::size_t hash<::FlexFlow::positive_int>::operator()(
    FlexFlow::positive_int n) const noexcept {
  return std::hash<int>{}(n.int_from_positive_int());
}

} // namespace std
