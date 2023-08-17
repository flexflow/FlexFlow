/* Copyright 2022 NVIDIA CORPORATION
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

#include "pool2d.h"

namespace triton { namespace backend { namespace legion {

// FIXME below are stub functions, need to fill in implementation
Pool2D::Pool2D(
    LegionModelState* model, const LayerStrategy* strategy, int kernelH,
    int kernelW, int strideH, int strideW, int paddingH, int paddingW,
    PoolType type, ActivationMode act, const char* name)
    : Operator(model, strategy, OperatorType::OP_POOL2D, name, 1, 0, 1),
      activation(act), pool_type(type), kernel_h(kernelH), kernel_w(kernelW),
      stride_h(strideH), stride_w(strideW), padding_h(paddingH),
      padding_w(paddingW)
{
}

Pool2D::~Pool2D() {}
void
Pool2D::Load(Realm::Processor processor)
{
}
void
Pool2D::Free(Realm::Processor processor)
{
}
void
Pool2D::Configure(Tensor* input, Tensor* output)
{
}
void
Pool2D::initialize(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
}
void
Pool2D::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
}
void
Pool2D::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
}
Pool2DArgs::Pool2DArgs(void) {}

#ifdef LEGION_USE_CUDA
Pool2DArgs
Pool2D::initialize_gpu(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
    Legion::Runtime* runtime)
{
  return Pool2DArgs();
}
void
Pool2D::forward_gpu(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
    Legion::Runtime* runtime)
{
}
void
Pool2D::forward_kernel(
    const Pool2DArgs* args, const void* input_ptr, void* output_ptr)
{
}
#endif

}}}  // namespace triton::backend::legion
