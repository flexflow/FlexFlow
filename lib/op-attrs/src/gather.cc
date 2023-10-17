#include "op-attrs/ops/gather.h"
#include "utils/exception.decl.h"
#include "utils/exceptions.h"

namespace FlexFlow {

bool GatherAttrs::is_valid(ParallelTensorShape const &lhs,
                           ParallelTensorShape const &rhs) const {
  if (lhs.dims.num_dims() != rhs.dims.num_dims()) {
    return false;
  }
  for (auto i : lhs.dims) {
    if (ff_dim_t(i.size) != this->dim &&
        lhs.at(ff_dim_t(i.size)).size < rhs.at(ff_dim_t(i.size)).size) {
      return false;
    }
  }
  return true;
}

//https://pytorch.org/docs/stable/generated/torch.gather.html
// todo: why return a vector?
std::vector<ParallelTensorShape>
    get_output_shapes(GatherAttrs const &attrs,
                      ParallelTensorShape const & input,
                      ParallelTensorShape const &index) {
  if(input.num_dims() != index.num_dims()) {
    throw mk_runtime_error("Gather: input and index must have the same number of dimensions");
  }

  for(int i = 0; i < input.num_dims(); i++) {
    if(i != attrs.dim && input.at(ff_dim_t(i)).size <= index.at(ff_dim_t(i)).size) {
      throw mk_runtime_error("Gather: index.size(d) <= input.size(d) for all dimensions d != dim");
    }
  }

  ParallelTensorShape output = input;

  std::vector<ParallelTensorShape> results;
  //NOTE(lambda):why return a vector?
  results.push_back(output);
  return results;
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
