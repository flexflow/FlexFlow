#include "flexflow/ops/conv_2d.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {

void Conv2D::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->out_channels);
  sez.serialize(this->kernel_h);
  sez.serialize(this->kernel_w);
  sez.serialize(this->stride_h);
  sez.serialize(this->stride_w);
  sez.serialize(this->padding_h);
  sez.serialize(this->padding_w);
  sez.serialize(this->groups);
  sez.serialize(this->use_bias);
  sez.serialize(this->activation);
}

using PCG::Node;
/*static*/
Node Conv2D::deserialize(FFModel& ff, Legion::Deserializer& dez, Tensor inputs[], int num_inputs) {
  assert (num_inputs == 1);

  int out_channels, kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w, groups;
  bool use_bias;
  ActiMode activation;

  dez.deserialize(out_channels);
  dez.deserialize(kernel_h);
  dez.deserialize(kernel_w);
  dez.deserialize(stride_h);
  dez.deserialize(stride_w);
  dez.deserialize(padding_h);
  dez.deserialize(padding_w);
  dez.deserialize(groups);
  dez.deserialize(use_bias);
  dez.deserialize(activation);

  return ff.get_or_create_conv2d_node(
      inputs[0],
      out_channels,
      kernel_h, kernel_w,
      stride_h, stride_w,
      padding_h, padding_w,
      activation,
      groups,
      use_bias);
}

}; // namespace FlexFlow
