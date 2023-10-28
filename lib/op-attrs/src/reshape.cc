#include "op-attrs/ops/reshape.h"
#include "op-attrs/ff_dim.h"
#include "op-attrs/parallel_tensor_shape.h"
#include "utils/exception.h"

namespace FlexFlow {

// https://pytorch.org/docs/stable/generated/torch.reshape.html
ParallelTensorShape get_output_shape(ReshapeAttrs const &attrs,
                                     ParallelTensorShape const &input) {
  std::size_t input_volume =
      input.dims.get_volume() / input.at(ff_dim_t(0)).size;
  std::size_t attrs_volume = 1;
  for (int i = 0; i < attrs.shape.dims.num_dims(); i++) {
    attrs_volume *= attrs.shape.at(ff_dim_t(i));
  }
  if (input_volume != attrs_volume) {
    throw mk_runtime_error("ReshapeAttrs: input_volume != attrs_volume");
  }

  std::vector<ParallelDim> data;

  if (attrs.shape.dims.num_dims() == 1) {
    // infer the shape
    if (attrs.shape.at(ff_dim_t(0)) == -1) {
      // the output shape will be (<r, d1 , t>, <input_volume, d2, f>)
      data.resize(2);
      data[0] = input.at(ff_dim_t(0));
      data[1].size = input_volume;
      // how to decide the degree?
      ParallelTensorShape output = ParallelTensorShape(
          ParallelTensorDims(TensorDims(data.begin(), data.end())),
          input.data_type);
      return output;
    } else {
      // i = attrs.shape.at(ff_dim_t(0)
      // the output shape will be (<r, d1 , t>, <i,_, f>, <input_volume /i , _,
      // f>)
      data.resize(3);
      data[0] = input.at(ff_dim_t(0));
      data[1].size = attrs.shape.at(ff_dim_t(0));
      data[2].size = input_volume / attrs.shape.at(ff_dim_t(0));
      for (int i = 1; i < 3; i++) {
        // how to decide the degree?
        data[i].is_replica_dim = false;
      }
      ParallelTensorShape output = ParallelTensorShape(
          ParallelTensorDims(TensorDims(data.begin(), data.end())),
          input.data_type);
      return output;
    }
  }

  ParallelTensorDims dims{attrs.shape.dims};
  ParallelTensorShape output = {dims, input.data_type};

  return output;
}

} // namespace FlexFlow
