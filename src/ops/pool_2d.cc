#include "ops/pool_2d.h"
#include "legion/legion_utilities.h"

void Pool2D::serialize(Legion::Serializer& sez) const {
  sez.serialize(this->kernel_h);
  sez.serialize(this->kernel_w);
  sez.serialize(this->stride_h);
  sez.serialize(this->stride_w);
  sez.serialize(this->padding_h);
  sez.serialize(this->padding_w);
  sez.serialize(this->pool_type);
  sez.serialize(this->activation);
}

/*static*/
Node Pool2D::deserialize(FFModel& ff, Legion::Deserializer& dez, Tensor inputs[], int num_inputs) { 
  assert (num_inputs == 1);

  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;

  dez.deserialize(kernel_h);
  dez.deserialize(kernel_w);
  dez.deserialize(stride_h);
  dez.deserialize(stride_w);
  dez.deserialize(padding_h);
  dez.deserialize(padding_w);
  dez.deserialize(pool_type);
  dez.deserialize(activation);

  return ff.get_or_create_pool2d_node(
      inputs[0],
      kernel_h, kernel_w,
      stride_h, stride_w,
      padding_h, padding_w,
      pool_type,
      activation);
}
