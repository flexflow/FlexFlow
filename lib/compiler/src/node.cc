#include "node.h"
#include "utils/hash-utils.h"
#include "op-attrs/ffconst_utils.h"

namespace FlexFlow {
namespace PCG {

Node::Node(size_t guid,
           OperatorParameters const &op_params)
  : guid(guid), op_params(op_params) { }

typename Node::AsConstTuple Node::as_tuple() const {
  return std::tie(this->guid, this->op_params, this->original_guid);
}

bool operator==(Node const &lhs, Node const &rhs) {
  return lhs.as_tuple() == rhs.as_tuple();
}

bool operator!=(Node const &lhs, Node const &rhs) {
  return lhs.as_tuple() != rhs.as_tuple();
}

bool operator<(Node const &lhs, Node const &rhs) {
  return lhs.as_tuple() < rhs.as_tuple();
}

std::string Node::to_string() const {
  return get_operator_type_name(get_op_type(this->op_params));
}

}
}

namespace std {
size_t hash<FlexFlow::PCG::Node>::operator()(FlexFlow::PCG::Node const &n) const {
  return get_std_hash(n.as_tuple());
};
}
