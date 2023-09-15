#include "op-attrs/ops/element_binary.h"

namespace FlexFlow {

ParallelTensorShape get_output_shape(ElementBinaryAttrs const &attrs,
                                     ParallelTensorShape const &in1,
                                     ParallelTensorShape const &in2) {
  ParallelTensorShape output = in1.num_dims() >= in2.num_dims() ? in1 : in2;
  for (int i = 0; i < output.num_dims(); i++) {
    if (i >= in1.num_dims()) {
      output.at(ff_dim_t(i)) = in2.at(ff_dim_t(i));
    } else if (i >= in2.num_dims()) {
      output.at(ff_dim_t(i)) = in1.at(ff_dim_t(i));
    } else if (in1.at(ff_dim_t(i)).size == in2.at(ff_dim_t(i)).size) {
      output.at(ff_dim_t(i)) = in1.at(ff_dim_t(i));
    } else if (in1.at(ff_dim_t(i)).size == 1) {
      output.at(ff_dim_t(i)) = in2.at(ff_dim_t(i));
    } else if (in2.at(ff_dim_t(i)).size == 1) {
      output.at(ff_dim_t(i)) = in1.at(ff_dim_t(i));
    } else {
      assert(false && "Operands could not be broadcast together");
      exit(0);
    }
  }

  return output;
}

} // namespace FlexFlow
