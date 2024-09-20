#include "pcg/create_grad.h"
#include "utils/exception.h"

namespace FlexFlow {

bool bool_from_create_grad(CreateGrad cg) {
  switch (cg) {
    case CreateGrad::YES:
      return true;
    case CreateGrad::NO:
      return false;
    default:
      throw mk_runtime_error(fmt::format("Unknown CreateGrad value {}", cg));
  }
}

} // namespace FlexFlow
