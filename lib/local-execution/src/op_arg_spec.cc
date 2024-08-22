#include "local-execution/op_arg_spec.h"

namespace FlexFlow {

std::type_index get_op_arg_spec_type_index(OpArgSpec const &s) {
  return s.visit<std::type_index>(
      [](auto &&arg) { return arg.get_type_index(); });
}

} // namespace FlexFlow
