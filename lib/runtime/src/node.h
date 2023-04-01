#ifndef _FLEXFLOW_RUNTIME_SRC_NODE_H
#define _FLEXFLOW_RUNTIME_SRC_NODE_H

#include <string>
#include "utils/optional.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

class Op;

struct Node {
  Node() = delete;
  Node(size_t _guid, Op const *_ptr);

  bool operator==(Node const &) const;
  bool operator!=(Node const &) const;
  bool operator<(Node const &) const;

  size_t guid;
  Op const *ptr;
  optional<size_t> original_guid = nullopt;
};

}

#endif
