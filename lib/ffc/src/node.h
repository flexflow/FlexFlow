#ifndef _FLEXFLOW_FFC_NODE_H
#define _FLEXFLOW_FFC_NODE_H

#include <string>

#include "tl/optional.hpp"
#include "op-meta/op-meta.h"

namespace FlexFlow {
namespace PCG {

struct Node {
  Node() = delete;
  Node(size_t guid, OperatorParameters const &op_params);

  bool operator==(Node const &b) const;
  bool operator!=(Node const &b) const;
  bool operator<(Node const &b) const;
  std::string to_string(void) const;

  size_t guid;
  OperatorParameters op_params;

  tl::optional<size_t> original_guid = tl::nullopt;

  using AsTuple = std::tuple<size_t &, OperatorParameters &>;
  using AsConstTuple = std::tuple<size_t const &, OperatorParameters const &>;

  AsTuple as_tuple();
  AsConstTuple as_tuple() const;
};

struct NodeCompare {
  bool operator()(Node const &a, Node const &b) const {
    if (a.guid != b.guid) {
      return a.guid < b.guid;
    }
    return a.op_params < b.op_params;
  };
};

}
}

#endif 
