#include "ops.h"
#include "inception.h"

void CnnModel::add_layers()
{
  printf("Create Resnet-121:\n");
  Tensor t = add_conv_layer(input_image, 64, 7, 7, 2, 2, 3, 3);
  t = add_pool_layer(t, 3, 3, 2, 2, 1, 1);
  for (int i = 0; i < 3; i++)
    t = BottleneckBlock(*this, t, 256, 64, 1);
  for (int i = 0; i < 4; i++) {
    int stride = (i==0) ? 2 : 1;
    t = BottleneckBlock(*this, t, 512, 128, stride);
  }
  for (int i = 0; i < 23; i++) {
    int stride = (i==0) ? 2 : 1;
    t = BottleneckBlock(*this, t, 1024, 256, stride);
  }
  for (int i = 0; i < 3; i++) {
    int stride = (i==0) ? 2 : 1;
    t = BottleneckBlock(*this, t, 2048, 512, stride);
  }
  t = add_pool_layer(t, 7, 7, 1, 1, 0, 0, POOL2D_AVG);
  t = add_flat_layer(t);
  t = add_linear_layer(t, 1000, false/*relu*/);
  t = add_softmax_layer(t);
}
