#include "node.h"

namespace FlexFlow {
namespace PCG {

Node::Node(size_t guid,
           OperatorParameters const &op_params)
  : guid(guid), op_params(op_params) { }

bool Node::operator==(Node const &b) const {
  return this->as_tuple() == b.as_tuple();
}

bool Node::operator!=(Node const &b) const {
  return this->as_tuple() != b.as_tuple();
}

bool Node::operator<(Node const &b) const {
  return this->as_tuple() < b.as_tuple();
}

std::string to_string(void) const {
  if (ptr != NULL) {
    return get_operator_type_name() + "_" + std::to_string(guid);
  } else {
    return "UnmappedOp_" + std::to_string(guid);
  }
}

}
}
