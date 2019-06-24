#include "model.h"

void FFModel::add_layers()
{
  printf("inputImage.numDim = %d\n", inputImage.numDim);
  Tensor t = conv2d("conv1", inputImage, 64, 11, 11, 4, 4, 2, 2);
  t = pool2d("pool1", t, 3, 3, 2, 2, 0, 0);
  t = conv2d("conv2", t, 192, 5, 5, 1, 1, 2, 2);
  t = pool2d("pool2", t, 3, 3, 2, 2, 0, 0);
  t = conv2d("conv3", t, 384, 3, 3, 1, 1, 1, 1);
  t = conv2d("conv4", t, 256, 3, 3, 1, 1, 1, 1);
  t = conv2d("conv5", t, 256, 3, 3, 1, 1, 1, 1);
  t = pool2d("pool3", t, 3, 3, 2, 2, 0, 0);
  t = flat("flat", t);
  t = linear("lienar1", t, 4096);
  t = linear("linear2", t, 4096);
  t = linear("linear3", t, 1000, false/*relu*/);
  t = softmax("softmax", t);
}
