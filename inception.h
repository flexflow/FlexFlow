/* Copyright 2017 Stanford, NVIDIA
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include "model.h"

Tensor InceptionA(CnnModel &model, Tensor input, int pool_features)
{
  Tensor t1 = model.add_conv_layer(input, 64, 1, 1, 1, 1, 0, 0);
  Tensor t2 = model.add_conv_layer(input, 48, 1, 1, 1, 1, 0, 0);
  t2 = model.add_conv_layer(t2, 64, 5, 5, 1, 1, 2, 2);
  Tensor t3 = model.add_conv_layer(input, 64, 1, 1, 1, 1, 0, 0);
  t3 = model.add_conv_layer(t3, 96, 3, 3, 1, 1, 1, 1);
  t3 = model.add_conv_layer(t3, 96, 3, 3, 1, 1, 1, 1);
  Tensor t4 = model.add_pool_layer(input, 3, 3, 1, 1, 1, 1, POOL2D_AVG);
  t4 = model.add_conv_layer(t4, pool_features, 1, 1, 1, 1, 0, 0);
  Tensor concat[4];
  concat[0] = t1; concat[1] = t2; concat[2] = t3; concat[3] = t4;
  Tensor output = model.add_concat_layer(4, concat);
  return output;
}

Tensor InceptionB(CnnModel &model, Tensor input)
{
  Tensor t1 = model.add_conv_layer(input, 384, 3, 3, 2, 2, 0, 0);
  Tensor t2 = model.add_conv_layer(input, 64, 1, 1, 1, 1, 0, 0);
  t2 = model.add_conv_layer(t2, 96, 3, 3, 1, 1, 1, 1);
  t2 = model.add_conv_layer(t2, 96, 3, 3, 2, 2, 0, 0);
  Tensor t3 = model.add_pool_layer(input, 3, 3, 2, 2, 0, 0);
  Tensor concat[3];
  concat[0] = t1; concat[1] = t2; concat[2] = t3;
  Tensor output = model.add_concat_layer(3, concat);
  return output;
}

Tensor InceptionC(CnnModel &model, Tensor input, int channels)
{
  Tensor t1 = model.add_conv_layer(input, 192, 1, 1, 1, 1, 0, 0);
  Tensor t2 = model.add_conv_layer(input, channels, 1, 1, 1, 1, 0, 0);
  t2 = model.add_conv_layer(t2, channels, 1, 7, 1, 1, 0, 3);
  t2 = model.add_conv_layer(t2, 192, 7, 1, 1, 1, 3, 0);
  Tensor t3 = model.add_conv_layer(input, channels, 1, 1, 1, 1, 0, 0);
  t3 = model.add_conv_layer(t3, channels, 7, 1, 1, 1, 3, 0);
  t3 = model.add_conv_layer(t3, channels, 1, 7, 1, 1, 0, 3);
  t3 = model.add_conv_layer(t3, channels, 7, 1, 1, 1, 3, 0);
  t3 = model.add_conv_layer(t3, 192, 1, 7, 1, 1, 0, 3);
  Tensor t4 = model.add_pool_layer(input, 3, 3, 1, 1, 1, 1, POOL2D_AVG);
  t4 = model.add_conv_layer(t4, 192, 1, 1, 1, 1, 0, 0);
  Tensor concat[4];
  concat[0] = t1; concat[1] = t2; concat[2] = t3; concat[3] = t4;
  Tensor output = model.add_concat_layer(4, concat);
  return output;
}

Tensor InceptionD(CnnModel &model, Tensor input)
{
  Tensor t1 = model.add_conv_layer(input, 192, 1, 1, 1, 1, 0, 0);
  t1 = model.add_conv_layer(t1, 320, 3, 3, 2, 2, 0, 0);
  Tensor t2 = model.add_conv_layer(input, 192, 1, 1, 1, 1, 0, 0);
  t2 = model.add_conv_layer(t2, 192, 1, 7, 1, 1, 0, 3);
  t2 = model.add_conv_layer(t2, 192, 7, 1, 1, 1, 3, 0);
  t2 = model.add_conv_layer(t2, 192, 3, 3, 2, 2, 0, 0);
  Tensor t3 = model.add_pool_layer(input, 3, 3, 2, 2, 0, 0);
  Tensor concat[3];
  concat[0] = t1; concat[1] = t2; concat[2] = t3;
  Tensor output = model.add_concat_layer(3, concat);
  return output;
}

Tensor InceptionE(CnnModel &model, Tensor input)
{
  Tensor t1 = model.add_conv_layer(input, 320, 1, 1, 1, 1, 0, 0);
  Tensor t2i = model.add_conv_layer(input, 384, 1, 1, 1, 1, 0, 0);
  Tensor t2 = model.add_conv_layer(t2i, 384, 1, 3, 1, 1, 0, 1);
  Tensor t3 = model.add_conv_layer(t2i, 384, 3, 1, 1, 1, 1, 0);
  Tensor t3i = model.add_conv_layer(input, 448, 1, 1, 1, 1, 0, 0);
  t3i = model.add_conv_layer(t3i, 384, 3, 3, 1, 1, 1, 1);
  Tensor t4 = model.add_conv_layer(t3i, 384, 1, 3, 1, 1, 0, 1);
  Tensor t5 = model.add_conv_layer(t3i, 384, 3, 1, 1, 1, 1, 0);
  Tensor t6 = model.add_pool_layer(input, 3, 3, 1, 1, 1, 1, POOL2D_AVG);
  t6 = model.add_conv_layer(t6, 192, 1, 1, 1, 1, 0, 0);
  Tensor concat[6];
  concat[0] = t1; concat[1] = t2; concat[2] = t3;
  concat[3] = t4; concat[4] = t5; concat[5] = t6;
  Tensor output = model.add_concat_layer(6, concat);
  return output;
}

Tensor DenseBlock(CnnModel &model, Tensor input, int numLayers, int growthRate)
{
  Tensor t, last = input;
  for (int i = 0; i < numLayers; i++) {
    t = model.add_bn_layer(last, true/*relu*/);
    t = model.add_conv_layer(t, 4 * growthRate, 1, 1, 1, 1, 0, 0, false/*relu*/);
    t = model.add_bn_layer(t, true/*relu*/);
    t = model.add_conv_layer(t, growthRate, 3, 3, 1, 1, 1, 1, false/*relu*/);
    Tensor concat[2];
    concat[0] = last; concat[1] = t;
    last = model.add_concat_layer(2, concat);
  }
  return last;
}

Tensor Transition(CnnModel &model, Tensor input, int outputSize)
{
  Tensor t = model.add_conv_layer(input, outputSize, 1, 1, 1, 1, 0, 0);
  t = model.add_pool_layer(t, 2, 2, 2, 2, 0, 0, POOL2D_AVG);
  return t;
}

Tensor BottleneckBlock(CnnModel &model, Tensor input, int outChannels,
                       int bnChannels, int stride)
{
  Tensor t = model.add_conv_layer(input, bnChannels, 1, 1, 1, 1, 0, 0);
  //t = model.add_bn_layer(t);
  t = model.add_conv_layer(t, bnChannels, 3, 3, stride, stride, 1, 1);
  //t = model.add_bn_layer(t);
  t = model.add_conv_layer(t, outChannels, 1, 1, 1, 1, 0, 0);
  //t = model.add_bn_layer(t);
  return t;
}
