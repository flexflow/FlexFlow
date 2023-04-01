#ifndef _FLEXFLOW_RUNTIME_SRC_NODE_H
#define _FLEXFLOW_RUNTIME_SRC_NODE_H

#include <string>
#include "utils/optional.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

class Op;

struct OpNode {
  OpNode() = delete;
  OpNode(size_t _guid, Op const *_ptr);

  bool operator==(OpNode const &) const;
  bool operator!=(OpNode const &) const;
  bool operator<(OpNode const &) const;

  size_t guid;
  Op const *ptr;
  optional<size_t> original_guid = nullopt;
};

}

#endif
