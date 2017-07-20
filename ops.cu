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

CnnHandle init_cudnn(const Task *task,
                     const std::vector<PhysicalRegion> &regions,
                     Context ctx, HighLevelRuntime *runtime)
{
  assert(regions.size() == 0);
  CnnHandle handle;
  checkCUDA(cublasCreate(&handle.blas));
  checkCUDNN(cudnnCreate(&handle.dnn));
  return handle;
}

Op::Op(Op* _pre_op)
{
  pre_ops.push_back(_pre_op);
  inputs.push_back(_pre_op.output);
}

Op::Op(std::vector<Op*> _pre_ops)
: pre_ops(_pre_ops)
{
  for (int i = 0; i < pre_ops.size(); i++) {
    inputs.push_back(pre_ops[i].output);
  }
}
