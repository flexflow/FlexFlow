#include "ops.h"
#include "inception.h"

void CnnModel::add_layers()
{
  Tensor t = add_conv_layer(input_image, 32, 3, 3, 2, 2, 0, 0);
  t = add_conv_layer(t, 32, 3, 3, 1, 1, 0, 0);
  t = add_conv_layer(t, 64, 3, 3, 1, 1, 1, 1);
  t = add_pool_layer(t, 3, 3, 2, 2, 0, 0);
  t = add_conv_layer(t, 80, 1, 1, 1, 1, 0, 0);
  t = add_conv_layer(t, 192, 3, 3, 1, 1, 1, 1);
  t = add_pool_layer(t, 3, 3, 2, 2, 0, 0);
  t = InceptionA(*this, t, 32);
  t = InceptionA(*this, t, 64);
  t = InceptionA(*this, t, 64);
  t = InceptionB(*this, t);
  t = InceptionC(*this, t, 128);
  t = InceptionC(*this, t, 160);
  t = InceptionC(*this, t, 160);
  t = InceptionC(*this, t, 192);
  t = InceptionD(*this, t);
  t = InceptionE(*this, t);
  t = InceptionE(*this, t);
  t = add_pool_layer(t, 8, 8, 1, 1, 0, 0, POOL2D_AVG);
  t = add_flat_layer(t);
  t = add_linear_layer(t, 1000, false/*relu*/);
  t = add_softmax_layer(t);
}
