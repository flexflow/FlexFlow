#include "pcg/file_format/v1/create_grad.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1CreateGrad to_v1(CreateGrad const &cg) {
  switch (cg) {
    case CreateGrad::YES:
      return V1CreateGrad::YES;
    case CreateGrad::NO:
      return V1CreateGrad::NO;
    default:
      NOT_REACHABLE();
  }
}

CreateGrad from_v1(V1CreateGrad const &vcg) {
  switch (vcg) {
    case V1CreateGrad::YES:
      return CreateGrad::YES;
    case V1CreateGrad::NO:
      return CreateGrad::NO;
    default:
      NOT_REACHABLE();
  }
}

} // namespace FlexFlow
