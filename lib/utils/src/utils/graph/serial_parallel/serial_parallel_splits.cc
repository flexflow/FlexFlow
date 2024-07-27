#include "utils/graph/serial_parallel/serial_parallel_splits.h"
#include "utils/fmt/unordered_set.h"
#include "utils/fmt/variant.h"
#include "utils/fmt/vector.h"
#include "utils/hash-utils.h"
#include "utils/hash/unordered_set.h"
#include "utils/hash/vector.h"

namespace FlexFlow {

SerialSplit::SerialSplit(
    std::vector<std::variant<ParallelSplit, Node>> const &children)
    : children(children) {}

SerialSplit::SerialSplit(
    std::initializer_list<std::variant<ParallelSplit, Node>> const &children)
    : children(children) {}

bool SerialSplit::operator==(SerialSplit const &other) const {
  return this->tie() == other.tie();
}

bool SerialSplit::operator!=(SerialSplit const &other) const {
  return this->tie() != other.tie();
}

SerialSplit::Tie SerialSplit::tie() const {
  return std::tie(this->children);
}

std::string format_as(SerialSplit const &split) {
  return fmt::format("<SerialSplit children={}>", split.children);
}

std::ostream &operator<<(std::ostream &s, SerialSplit const &split) {
  return s << fmt::to_string(split);
}

ParallelSplit::ParallelSplit(
    std::unordered_set<std::variant<SerialSplit, Node>> const &children)
    : children(children) {}

ParallelSplit::ParallelSplit(
    std::initializer_list<std::variant<SerialSplit, Node>> const &children)
    : children(children) {}

bool ParallelSplit::operator==(ParallelSplit const &other) const {
  return this->tie() == other.tie();
}

bool ParallelSplit::operator!=(ParallelSplit const &other) const {
  return this->tie() != other.tie();
}

ParallelSplit::Tie ParallelSplit::tie() const {
  return std::tie(this->children);
}

std::string format_as(ParallelSplit const &split) {
  return fmt::format("<ParallelSplit children={}>", split.children);
}

std::ostream &operator<<(std::ostream &s, ParallelSplit const &split) {
  return s << fmt::to_string(split);
}

} // namespace FlexFlow

namespace std {

size_t hash<::FlexFlow::SerialSplit>::operator()(
    ::FlexFlow::SerialSplit const &s) const {
  size_t result = 0;
  ::FlexFlow::hash_combine(result, s.children);
  return result;
}

size_t hash<::FlexFlow::ParallelSplit>::operator()(
    ::FlexFlow::ParallelSplit const &s) const {
  size_t result = 0;
  ::FlexFlow::hash_combine(result, s.children);
  return result;
}

} // namespace std
