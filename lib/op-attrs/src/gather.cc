#include "op-attrs/ops/gather.h"
#include "utils/exception.h"

namespace FlexFlow {

// https://pytorch.org/docs/stable/generated/torch.gather.html
//  todo: why return a vector?
std::vector<ParallelTensorShape>
    get_output_shapes(GatherAttrs const &attrs,
                      ParallelTensorShape const &input,
                      ParallelTensorShape const &index) {
  if (input.num_dims() != index.num_dims()) {
    throw mk_runtime_error(
        "for gather, the dimensions of input and index are not match");
  }

  for (int i = 1; i < input.num_dims(); i++) {
    if (i != attrs.dim &&
        input.at(ff_dim_t(i)).size <= index.at(ff_dim_t(i)).size) {
      throw mk_runtime_error(
          "Gather: index.size(d) <= input.size(d) for all dimensions d != dim");
    }

    ParallelTensorShape output = index;
    output.at(ff_dim_t(0)) = input.at(ff_dim_t(0));
    std::vector<ParallelTensorShape> results;
    // NOTE(lambda):why return a vector?
    results.push_back(output);
    return results;
  }
}
/* bool GatherAttrs::is_valid(ParallelTensorShape const &lhs,
 * ParallelTensorShape const &rhs) const { */
/*   if (lhs.num_dims() != rhs.num_dims()) { */
/*     return false; */
/*   } */
/*   for (int i = 0; i < lhs.num_dims(); i++) { */
/*     if (i != this->legion_dim && */
/*         lhs.at(i).size < rhs.at(i).size) { */
/*       return false; */
/*     } */
/*   } */
/*   return true; */
/* } */

} // namespace FlexFlow
