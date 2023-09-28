#include "pcg/file_format/v1/param_sync.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1ParamSync to_v1(ParamSync const &p) {
  switch (p) {
    case ParamSync::PS:
      return V1ParamSync::PARAM_SERVER;
    case ParamSync::NCCL:
      return V1ParamSync::NCCL;
    default:
      NOT_IMPLEMENTED();
  };
}

} // namespace FlexFlow
