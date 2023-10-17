#include "op-attrs/ops/reshape.h"
#include "op-attrs/ff_dim.h"
#include "utils/exception.h"

namespace FlexFlow {

//https://pytorch.org/docs/stable/generated/torch.reshape.html
// pytorch: the input: [2,3,4], shape maybe [-1,6]， should we add this? and the
// output is [4, 6] currently we doesn't consider the case of -1,we can support
// this later the input:[2,3,4], attrs.shape:[4,6], the output is [4, 6]
ParallelTensorShape get_output_shape(ReshapeAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  std::size_t input_volume = input.dims.get_volume();
  std::size_t attrs_volume = 1;
  for (int i = 0; i < attrs.shape.dims.num_dims(); i++) {
    attrs_volume *= attrs.shape.at(ff_dim_t(i));
  }
  if(input_volume != attrs_volume) {
    throw mk_runtime_error("ReshapeAttrs: input_volume != attrs_volume");
  }

  ParallelTensorShape output = input;
  output.data_type = input.data_type;
  if(attrs.shape.dims.num_dims() == 1) {
      //infer the shape
      if(attrs.shape.at(ff_dim_t(0)) == -1) {
       
        output.at(ff_dim_t(0)).size = input_volume ;
        output.at(ff_dim_t(0)).degree = 1;
        output.at(ff_dim_t(0)).is_replica_dim = false;
      } else {
        output.at(ff_dim_t(0)).size = attrs.shape.at(ff_dim_t(0));
        output.at(ff_dim_t(1)).size = input_volume / attrs.shape.at(ff_dim_t(0));
        for(int i = 0; i < 2; i++) {
          output.at(ff_dim_t(i)).degree = 1;
          output.at(ff_dim_t(i)).is_replica_dim = false;
        }
      }
  } else {
      ParallelTensorDims dims{attrs.shape.dims};
      output = {dims, input.data_type};
  }
  return output;
}

} // namespace FlexFlow
