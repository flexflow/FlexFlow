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
#include "ops.h"

CnnHandle init_cudnn(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 0);
  assert(task->arglen == sizeof(size_t));
  size_t workSpaceSize = *(const size_t*) task->args;
  CnnHandle handle;
  handle.workSpaceSize = workSpaceSize;
  checkCUDA(cublasCreate(&handle.blas));
  checkCUDNN(cudnnCreate(&handle.dnn));
  checkCUDA(cudaMalloc(&handle.workSpace, workSpaceSize));
  return handle;
}

Op::Op(Tensor input)
{
  inputs[0] = input;
}

