#include "pcg/file_format/v1/tensor.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1Tensor to_v1(Tensor const &t) {
  return {
      to_v1(t.get_shape()),
      t.create_gradients,
      to_v1<V1Initializer>(t.initializer),
      to_v1<V1ParamSync>(t.sync_type),
      t.name,
  };
}

} // namespace FlexFlow
