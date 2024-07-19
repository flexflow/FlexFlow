#include "utils/containers/without_nullopts.h"

namespace FlexFlow {

template std::unordered_set<int>
    without_nullopts(std::unordered_set<std::optional<int>> const &);
template std::vector<int>
    without_nullopts(std::vector<std::optional<int>> const &);

} // namespace FlexFlow
