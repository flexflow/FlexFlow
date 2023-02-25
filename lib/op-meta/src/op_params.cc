#include "op-meta/ops/op_params.h"

namespace FlexFlow {
namespace opmeta {

int OpParamsInterface::num_outputs(std::vector<ParallelTensorShape> const &inputs) const {
  return this->output_shapes(inputs).size();
}

}
}
