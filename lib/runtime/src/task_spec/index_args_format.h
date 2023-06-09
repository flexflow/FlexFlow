#ifndef _FLEXFLOW_RUNTIME_SRC_INDEX_ARGS_FORMAT_H
#define _FLEXFLOW_RUNTIME_SRC_INDEX_ARGS_FORMAT_H

#include "concrete_args_format.h"
#include "legion.h"
#include "tensorless_task_invocation.h"
#include <map>

namespace FlexFlow {

struct IndexArgsFormat : public use_visitable_cmp<IndexArgsFormat> {
  IndexArgsFormat() = delete;
  IndexArgsFormat(
      std::map<Legion::DomainPoint, ConcreteArgsFormat> const &point_map)
      : point_map(point_map) {}

public:
  std::map<Legion::DomainPoint, ConcreteArgsFormat> point_map;
};

IndexArgsFormat process_index_args(TensorlessTaskBinding const &,
                                   Legion::Domain const &);
ConcreteArgsFormat process_index_args_for_point(
    std::unordered_map<slot_id, IndexArgSpec> const &,
    Legion::DomainPoint const &);
ConcreteArgSpec resolve_index_arg(IndexArgSpec const &,
                                  Legion::DomainPoint const &);

} // namespace FlexFlow

#endif
