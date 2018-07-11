/* Copyright 2018 Stanford
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

#ifndef _FLEXFLOW_CONFIG_H_
#define _FLEXFLOW_CONFIG_H_
#include <cstring>
#include "legion.h"
class FFConfig {
public:
  int epoches, batchSize, inputHeight, inputWidth;
  int numNodes, loadersPerNode, workersPerNode;
  float learningRate, weightDecay;
  size_t workSpaceSize;
  Context lg_ctx;
  Runtime* lg_hlr;
  bool syntheticInput;
  std::string datasetPath, strategyFile;
};
#endif//_FLEXFLOW_CONFIG_H_
