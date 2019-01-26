#include "ops.h"

void CnnModel::add_layers()
{
  Tensor t = add_conv_layer(input_image, 64, 11, 11, 4, 4, 2, 2);
  t = add_pool_layer(t, 3, 3, 2, 2, 0, 0);
  t = add_conv_layer(t, 192, 5, 5, 1, 1, 2, 2);
  t = add_pool_layer(t, 3, 3, 2, 2, 0, 0);
  t = add_conv_layer(t, 384, 3, 3, 1, 1, 1, 1);
  t = add_conv_layer(t, 256, 3, 3, 1, 1, 1, 1);
  t = add_conv_layer(t, 256, 3, 3, 1, 1, 1, 1);
  t = add_pool_layer(t, 3, 3, 2, 2, 0, 0);
  t = add_flat_layer(t);
  t = add_linear_layer(t, 4096);
  t = add_linear_layer(t, 4096);
  t = add_linear_layer(t, 1000, false/*relu*/);
  t = add_softmax_layer(t);
}
