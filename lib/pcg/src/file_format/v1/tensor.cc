#include "pcg/file_format/v1/tensor.h"
#include "pcg/file_format/v1/v1.h"

namespace FlexFlow {

V1Tensor to_v1(Tensor const &t) {
  return {to_v1(t.get_shape()),
          to_v1(t.create_gradients),
          to_v1<V1Initializer>(t.initializer),
          to_v1<V1ParamSync>(t.sync_type),
          t.name};
}

Tensor from_v1(V1Tensor const &vt) {
  TensorShape shape = from_v1(vt.shape);
  return {shape.dims,
          shape.data_type,
          from_v1(vt.create_gradients),
          from_v1<Initializer>(vt.initializer),
          from_v1<ParamSync>(vt.sync_type),
          vt.name};
}

} // namespace FlexFlow
