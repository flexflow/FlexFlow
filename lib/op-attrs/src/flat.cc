#include "op-attrs/ops/flat.h"
#include "op-attrs/ff_dim.h"
#include "parallel_dim_mapping_record.h"
#include "parallel_dim_mapping_record_solver.h"
#include "utils/exception.h"
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
// input: (<ri, di, t>, <x0, d1, f>, <x1,d2, f>, ......)
// assume d1=d2=d3
// output: 2d dimention (<ri, di, t>, <x0+x1+x2+x3, d0, f> )
ParallelTensorShape get_output_shape(FlatAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  if (input.num_dims() < 2) {
    throw mk_runtime_error("for flat,its dims must greater than 2");
  }

  int degree = input.at(ff_dim_t(1)).degree;
  for (int i = 1; i < input.num_dims(); i++) {
    if (degree != input.at(ff_dim_t(i)).degree) {
      throw mk_runtime_error(
          "for flat, all degree should be equal, but elemement ", i, " not");
    }
  }
  std::vector<ParallelDim> data;
  data.resize(2);
  data[0] = input.at(ff_dim_t(0));
  data[0].is_replica_dim = true;
  data[1].degree = input.at(ff_dim_t(1)).degree;
  data[1].size = input.at(ff_dim_t(1)).size;
  data[1].is_replica_dim = false;

  for (int i = 2; i < input.num_dims(); i++) {
    data[1].size *= input.at(ff_dim_t(i)).size;
  }

  ParallelTensorShape output = ParallelTensorShape(
      ParallelTensorDims(TensorDims(data.begin(), data.end())),
      input.data_type);

  return output;
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
