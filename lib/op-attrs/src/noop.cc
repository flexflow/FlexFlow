#include "op-attrs/ops/noop.h"
#include "utils/visitable_funcs.h"

namespace FlexFlow {

bool operator==(NoopAttrs const &lhs, NoopAttrs const &rhs) {
  return true;
}

bool operator!=(NoopAttrs const &lhs, NoopAttrs const &rhs) {
  return false;
}

bool operator<(NoopAttrs const &lhs, NoopAttrs const &rhs) {
  return false;
}

InputAttrs::InputAttrs(std::size_t _input_tensor_guid)
  : input_tensor_guid(_input_tensor_guid)
{ }

bool operator==(InputAttrs const &lhs, InputAttrs const &rhs) {
  return visit_eq(lhs, rhs);
}

bool operator!=(InputAttrs const &lhs, InputAttrs const &rhs) {
  return visit_neq(lhs, rhs);
}

bool operator<(InputAttrs const &lhs, InputAttrs const &rhs) {
  return visit_lt(lhs, rhs);
}

}

namespace std {

using ::FlexFlow::NoopAttrs;
using ::FlexFlow::InputAttrs;

size_t hash<NoopAttrs>::operator()(NoopAttrs const &a) const {
  return 0;
}

size_t hash<InputAttrs>::operator()(InputAttrs const &a) const {
  return visit_hash(a);
}

}
