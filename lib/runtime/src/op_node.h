#ifndef _FLEXFLOW_RUNTIME_SRC_NODE_H
#define _FLEXFLOW_RUNTIME_SRC_NODE_H

#include <string>
#include "utils/optional.h"
#include "utils/visitable_funcs.h"
#include "utils/strong_typedef.h"
#include "op-attrs/ffconst.h"

namespace FlexFlow {

class Op;

struct op_node_guid_t : strong_typedef<op_node_guid_t, size_t> {
  using strong_typedef::strong_typedef;
};

struct OpNode {
  OpNode() = delete;
  OpNode(op_node_guid_t, Op const *);

  bool operator==(OpNode const &) const;
  bool operator!=(OpNode const &) const;
  bool operator<(OpNode const &) const;

  op_node_guid_t guid;
  Op const *ptr;
  optional<size_t> original_guid = nullopt;
};

struct OpNodeManager {
public:
  OpNode create(Op const *);
private:
  size_t op_node_guid = NODE_GUID_FIRST_VALID;
};

}

#endif
