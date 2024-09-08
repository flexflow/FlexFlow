#include "utils/graph/series_parallel/series_parallel_splits.h"
#include "utils/fmt/unordered_multiset.h"
#include "utils/fmt/variant.h"
#include "utils/fmt/vector.h"
#include "utils/hash-utils.h"
#include "utils/hash/unordered_multiset.h"
#include "utils/hash/vector.h"

namespace FlexFlow {

SeriesSplit::SeriesSplit(
    std::vector<std::variant<ParallelSplit, Node>> const &children)
    : children(children) {}

SeriesSplit::SeriesSplit(
    std::initializer_list<std::variant<ParallelSplit, Node>> const &children)
    : children(children) {}

bool SeriesSplit::operator==(SeriesSplit const &other) const {
  return this->tie() == other.tie();
}

bool SeriesSplit::operator!=(SeriesSplit const &other) const {
  return this->tie() != other.tie();
}

SeriesSplit::Tie SeriesSplit::tie() const {
  return std::tie(this->children);
}

std::string format_as(SeriesSplit const &split) {
  return fmt::format("<SeriesSplit children={}>", split.children);
}

std::ostream &operator<<(std::ostream &s, SeriesSplit const &split) {
  return s << fmt::to_string(split);
}

ParallelSplit::ParallelSplit(
    std::unordered_multiset<std::variant<SeriesSplit, Node>> const &children)
    : children(children) {}

ParallelSplit::ParallelSplit(
    std::initializer_list<std::variant<SeriesSplit, Node>> const &children)
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

size_t hash<::FlexFlow::SeriesSplit>::operator()(
    ::FlexFlow::SeriesSplit const &s) const {
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
