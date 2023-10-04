#include "pcg/file_format/v1/tensor_shape.h"

namespace FlexFlow {

V1TensorShape to_v1(TensorShape const &shape) {
  return {std::vector<size_t>(shape.dims.begin(), shape.dims.end()),
          to_v1(shape.data_type)};
}

} // namespace FlexFlow
