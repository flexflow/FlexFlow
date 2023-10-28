#include "op-attrs/ops/groupby.h"
#include "utils/exception.h"

namespace FlexFlow {

// import torch
// data = torch.tensor([10, 20, 30, 40, 50, 60, 70, 80])
// # group index tensor group_indices
// group_indices = torch.tensor([0, 1, 0, 2, 1, 2, 0, 1])

// # groupby operator
// unique_indices, unique_inverse_indices =
// torch.unique(group_indices,return_inverse=True)

// print(f"unique_indices: {unique_indices} and unique_inverse_indices:
// {unique_inverse_indices}")

// grouped_data = []

// for i in unique_indices: # use unique_inverse_indices
//     group_data = data[unique_inverse_indices == i]
//     grouped_data.append(group_data)

// for i, group in enumerate(grouped_data):
//     print(f"Group {i}: {group}")

// Group 0: tensor([10, 30, 70])
//  Group 1: tensor([20, 50, 80])
//  Group 2: tensor([40, 60])

ParallelTensorShape get_output_shape(Group_byAttrs const &attrs,
                                     ParallelTensorShape const &input_shape,
                                     ParallelTensorShape const &index) {
  if (input_shape.num_dims() != index.num_dims()) {
    throw mk_runtime_error(
        "Group_by: input and index must have the same number of dimensions");
  }
  // Note: how  to decide the groupby output shape?
  return input_shape;
}

} // namespace FlexFlow
