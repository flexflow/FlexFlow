#include "substitutions/tensor_pattern/get_attribute.h"
#include "utils/containers.h"

namespace FlexFlow {

TensorAttributeValue get_attribute(ParallelTensorAttrs const &attrs, TensorAttributeKey key) {
  switch (key) {
    case TensorAttributeKey::DIM_SIZES: {
      std::vector<size_t> sizes = transform(as_vector(ff_ordered(attrs.shape.dims)),
                  [](ParallelDim const &d) { return d.size; });
      return TensorAttributeValue{sizes};
    }
    case TensorAttributeKey::DIM_DEGREES: {
      std::vector<size_t> degrees = transform(as_vector(ff_ordered(attrs.shape.dims)),
                                              [](ParallelDim const &d) { 
                                                return static_cast<size_t>(d.degree); 
                                              });
      return TensorAttributeValue{degrees};
    }
    default:
      throw std::runtime_error(fmt::format("Unknown TensorAttributeKey {}", static_cast<int>(key)));
  }
}

} // namespace FlexFlow
