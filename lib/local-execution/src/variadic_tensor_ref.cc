#include "variadic_tensor_ref.h"

namespace FlexFlow {

VariadicTensorRef<OpTensorSpec> get_input_tensors() {
  return {VariadicTensorRefType::INPUT_TENSORS};
}

}