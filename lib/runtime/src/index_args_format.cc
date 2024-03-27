#include "index_args_format.h"

namespace FlexFlow {

IndexArgsFormat process_index_args(TensorlessTaskBinding const &binding,
                                   Legion::Domain const &domain) {
  std::map<Legion::DomainPoint, ConcreteArgsFormat> point_map;
  auto index_args = get_args_of_type<IndexArgSpec>(binding);
  for (Legion::Domain::DomainPointIterator it(domain); it; it++) {
    point_map.insert({*it, process_index_args_for_point(index_args, *it)});
  }
  return {point_map};
};

ConcreteArgsFormat process_index_args_for_point(
    std::unordered_map<slot_id, IndexArgSpec> const &specs,
    Legion::DomainPoint const &p) {
  std::unordered_map<slot_id, ConcreteArgSpec> resolved = map_values(
      specs, [&](IndexArgSpec const &s) { return resolve_index_arg(s, p); });
  return process_concrete_args(resolved);
}

} // namespace FlexFlow
