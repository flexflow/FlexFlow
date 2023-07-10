#ifndef _FLEXFLOW_FFC_NODE_H
#define _FLEXFLOW_FFC_NODE_H

#include <string>

#include "op-attrs/op-attrs.h"
#include "tl/optional.hpp"

namespace FlexFlow {
namespace ffc {

struct Node {
  Node() = delete;
  Node(size_t guid, PCGOperatorAttrs const &op_params);

  std::string to_string(void) const;

  using AsTuple =
      std::tuple<size_t &, PCGOperatorAttrs &, tl::optional<size_t> &>;
  using AsConstTuple = std::tuple<size_t const &,
                                  PCGOperatorAttrs const &,
                                  tl::optional<size_t> const &>;

  AsTuple as_tuple();
  AsConstTuple as_tuple() const;

public:
  size_t guid;
  PCGOperatorAttrs op_params;
  tl::optional<size_t> original_guid = tl::nullopt;
};

bool operator==(Node const &, Node const &);
bool operator!=(Node const &, Node const &);
bool operator<(Node const &, Node const &);

} // namespace ffc
} // namespace FlexFlow

namespace std {
template <>
struct hash<::FlexFlow::ffc::Node> {
  size_t operator()(::FlexFlow::ffc::Node const &n) const;
};
} // namespace std

#endif
