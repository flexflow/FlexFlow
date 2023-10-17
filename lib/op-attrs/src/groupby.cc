#include "op-attrs/ops/groupby.h"
#include "utils/exceptions.h"

namespace FlexFlow {

/*
import torch
data = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])
# group index tensor group_indices
group_indices = torch.tensor([0, 1, 0, 2, 1, 2, 0, 1])

# groupby operator
unique_indices, unique_inverse_indices = torch.unique(group_indices,
return_inverse=True) print(f"unique_indices: {unique_indices} and
unique_inverse_indices: {unique_inverse_indices}") grouped_data = [] for i in
unique_indices: # use unique_inverse_indices group_data =
data[unique_inverse_indices == i] grouped_data.append(group_data) for i, group
in enumerate(grouped_data): print(f"Group {i}: {group}")
*/

ParallelTensorShape get_output_shape(Group_byAttrs const &attrs,
                                     ParallelTensorShape const &input,
                                     ParallelTensorShape const &index) {
  if (input.num_dims() != index.num_dims()) {
    throw mk_runtime_error(
        "Group_by: input and index must have the same number of dimensions");
  }

  ParallelTensorShape output = input;
  // degree of output is same as input's
  return output;
}

} // namespace FlexFlow
