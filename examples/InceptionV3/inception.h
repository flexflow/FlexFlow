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

//#include "ops.h"

Tensor InceptionA(FFModel ff, Tensor input, int pool_features, std::string prefix = "")
{
  Tensor t1 = ff.conv2d(prefix + "conv1", input, 64, 1, 1, 1, 1, 0, 0);
  Tensor t2 = ff.conv2d(prefix + "ia_conv2", input, 48, 1, 1, 1, 1, 0, 0);
  t2 = ff.conv2d(prefix + "conv3",t2, 64, 5, 5, 1, 1, 2, 2);
  Tensor t3 = ff.conv2d(prefix + "conv4", input, 64, 1, 1, 1, 1, 0, 0);
  t3 = ff.conv2d(prefix + "conv5", t3, 96, 3, 3, 1, 1, 1, 1);
  t3 = ff.conv2d(prefix + "conv6", t3, 96, 3, 3, 1, 1, 1, 1);
  Tensor t4 = ff.pool2d(prefix + "pool1", input, 3, 3, 1, 1, 1, 1, POOL_AVG);
  t4 = ff.conv2d(prefix + "conv7", t4, pool_features, 1, 1, 1, 1, 0, 0);
  Tensor concat[4];
  concat[0] = t1; concat[1] = t2; concat[2] = t3; concat[3] = t4;
  Tensor output = ff.concat(prefix + "concat1", 4, concat, 1);

  return output;
}

Tensor InceptionB(FFModel ff, Tensor input, std::string prefix = "")
{
  Tensor t1 = ff.conv2d(prefix + "conv1", input, 384, 3, 3, 2, 2, 0, 0);
  Tensor t2 = ff.conv2d(prefix + "conv2", input, 64, 1, 1, 1, 1, 0, 0);
  t2 = ff.conv2d(prefix + "conv3", t2, 96, 3, 3, 1, 1, 1, 1);
  t2 = ff.conv2d(prefix + "conv4", t2, 96, 3, 3, 2, 2, 0, 0);
  Tensor t3 = ff.pool2d(prefix + "pool1", input, 3, 3, 2, 2, 0, 0);
  Tensor concat[3];
  concat[0] = t1; concat[1] = t2; concat[2] = t3;
  Tensor output = ff.concat(prefix + "concat1", 3, concat, 1);
  return output;
}

Tensor InceptionC(FFModel ff, Tensor input, int channels, std::string prefix = "")
{
  Tensor t1 = ff.conv2d(prefix + "conv1", input, 192, 1, 1, 1, 1, 0, 0);
  Tensor t2 = ff.conv2d(prefix + "conv2", input, channels, 1, 1, 1, 1, 0, 0);
  t2 = ff.conv2d(prefix + "conv3", t2, channels, 1, 7, 1, 1, 0, 3);
  t2 = ff.conv2d(prefix + "conv4", t2, 192, 7, 1, 1, 1, 3, 0);
  Tensor t3 = ff.conv2d(prefix + "pool1", input, channels, 1, 1, 1, 1, 0, 0);
  t3 = ff.conv2d(prefix + "conv5", t3, channels, 7, 1, 1, 1, 3, 0);
  t3 = ff.conv2d(prefix + "conv6", t3, channels, 1, 7, 1, 1, 0, 3);
  t3 = ff.conv2d(prefix + "conv7", t3, channels, 7, 1, 1, 1, 3, 0);
  t3 = ff.conv2d(prefix + "conv8", t3, 192, 1, 7, 1, 1, 0, 3);
  Tensor t4 = ff.pool2d(prefix + "pool2", input, 3, 3, 1, 1, 1, 1, POOL_AVG);
  t4 = ff.conv2d(prefix + "conv9", t4, 192, 1, 1, 1, 1, 0, 0);
  Tensor concat[4];
  concat[0] = t1; concat[1] = t2; concat[2] = t3; concat[3] = t4;
  Tensor output = ff.concat(prefix + "concat1", 4, concat, 1);
  return output;
}

Tensor InceptionD(FFModel ff, Tensor input, std::string prefix = "")
{
  Tensor t1 = ff.conv2d(prefix + "conv1", input, 192, 1, 1, 1, 1, 0, 0);
  t1 = ff.conv2d(prefix + "conv2", t1, 320, 3, 3, 2, 2, 0, 0);
  Tensor t2 = ff.conv2d(prefix + "conv3", input, 192, 1, 1, 1, 1, 0, 0);
  t2 = ff.conv2d(prefix + "conv4", t2, 192, 1, 7, 1, 1, 0, 3);
  t2 = ff.conv2d(prefix + "conv5", t2, 192, 7, 1, 1, 1, 3, 0);
  t2 = ff.conv2d(prefix + "conv6", t2, 192, 3, 3, 2, 2, 0, 0);
  Tensor t3 = ff.pool2d(prefix + "pool2", input, 3, 3, 2, 2, 0, 0);
  Tensor concat[3];
  concat[0] = t1; concat[1] = t2; concat[2] = t3;
  Tensor output = ff.concat(prefix + "concat1", 3, concat, 1);
  return output;
}

Tensor InceptionE(FFModel ff, Tensor input, std::string prefix = "")
{
  Tensor t1 = ff.conv2d(prefix + "conv1", input, 320, 1, 1, 1, 1, 0, 0);
  Tensor t2i = ff.conv2d(prefix + "conv2", input, 384, 1, 1, 1, 1, 0, 0);
  Tensor t2 = ff.conv2d(prefix + "conv3", t2i, 384, 1, 3, 1, 1, 0, 1);
  Tensor t3 = ff.conv2d(prefix + "conv4", t2i, 384, 3, 1, 1, 1, 1, 0);
  Tensor t3i = ff.conv2d(prefix + "conv5", input, 448, 1, 1, 1, 1, 0, 0);
  t3i = ff.conv2d(prefix + "conv6", t3i, 384, 3, 3, 1, 1, 1, 1);
  Tensor t4 = ff.conv2d(prefix + "conv7", t3i, 384, 1, 3, 1, 1, 0, 1);
  Tensor t5 = ff.conv2d(prefix + "conv8", t3i, 384, 3, 1, 1, 1, 1, 0);
  Tensor t6 = ff.pool2d(prefix + "conv9", input, 3, 3, 1, 1, 1, 1, POOL_AVG);
  t6 = ff.conv2d(prefix + "conv10", t6, 192, 1, 1, 1, 1, 0, 0);
  Tensor concat[6];
  concat[0] = t1; concat[1] = t2; concat[2] = t3;
  concat[3] = t4; concat[4] = t5; concat[5] = t6;
  Tensor output = ff.concat(prefix + "concat1", 6, concat, 1);
  return output;
}
