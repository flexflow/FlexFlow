#include "op-attrs/ops/flat.h"
#include "parallel_dim_mapping_record.h"
#include "parallel_dim_mapping_record_solver.h"
#include <cassert>

namespace FlexFlow {

namespace Input {
constexpr int NUMDIM = 5, WIDTH = 0, HEIGHT = 1, CHANNEL = 2, SAMPLE = 3,
              REPLICA = 4;
}

namespace Output {
constexpr int NUMDIM = 3, CHANNEL = 0, SAMPLE = 1, REPLICA = 2;
}

// flat is like the pytorch view
// tensor = torch.randn(2, 3, 4)  ,flattened_tensor = tensor.view(-1) #shape:
// (24)
ParallelTensorShape get_output_shape(FlatAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output_shape(input.dims, input.data_type);

  output_shape.at(ff_dim_t(Output::CHANNEL)).size =
      input.at(ff_dim_t(Input::CHANNEL)).size *
      input.at(ff_dim_t(Input::HEIGHT)).size *
      input.at(ff_dim_t(Input::WIDTH)).size;
  output_shape.at(ff_dim_t(Output::CHANNEL)).degree =
      input.at(ff_dim_t(Input::CHANNEL)).degree;
  output_shape.at(ff_dim_t(Output::CHANNEL)).is_replica_dim =
      (input.at(ff_dim_t(Input::CHANNEL)).degree > 1);

  return output_shape;
}

/* bool FlatAttrs::is_valid(ParallelTensorShape const &input) const { */
/*   ParallelTensorShape output_shape = this->calculate_output_shape(input); */

/*   bool is_valid = true; */
/*   is_valid &= input.is_valid(); */
/*   is_valid &= output_shape.is_valid(); */
/*   is_valid &= (input.at(Input::WIDTH).degree == 1); */

/*   return is_valid; */
/* } */

/* ParallelTensorShape FlatAttrs::calculate_output_shape(ParallelTensorShape
 * const &input) const { */
/*   assert (input.num_dims() == Input::NUMDIM); */
/*   ParallelTensorShape output_dims; */
/*   output_dims.data_type = input.data_type; */

/*   output_dims.at(Output::REPLICA) = input.at(Input::REPLICA); */
/*   output_dims.at(Output::SAMPLE) = input.at(Input::SAMPLE); */

/*   output_dims.at(Output::CHANNEL).degree = input.at(Input::CHANNEL).degree;
 */
/*   assert (input.at(Input::HEIGHT).degree == 1); */
/*   assert (input.at(Input::WIDTH).degree == 1); */

/*   output_dims.at(Output::CHANNEL).size = input.at(Input::CHANNEL).size *
 * input.at(Input::HEIGHT).size * input.at(Input::WIDTH).size; */
/*   output_dims.at(Output::CHANNEL).parallel_idx =
 * input.at(Input::CHANNEL).parallel_idx; */

/*   return output_dims; */
/* } */

} // namespace FlexFlow
