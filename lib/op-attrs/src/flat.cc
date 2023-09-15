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

ParallelTensorShape get_output_shape(FlatAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  ParallelTensorShape output_shape(input.dims, input.data_type);

  output_shape.at(ff_dim_t(Output::CHANNEL)).size =
      input.at(ff_dim_t(Input::CHANNEL)).size *
      input.at(ff_dim_t(Input::HEIGHT)).size *
      input.at(ff_dim_t(Input::WIDTH)).size;
  output_shape.at(ff_dim_t(Output::CHANNEL)).degree =
      input.at(ff_dim_t(Input::CHANNEL)).degree;

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

} // namespace FlexFlow
