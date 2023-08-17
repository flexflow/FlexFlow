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

#include "operators/binary.h"

namespace triton { namespace backend { namespace legion {

BinaryOperator::BinaryOperator(
    LegionModelState* model, const LayerStrategy* strategy, OperatorType type,
    bool inplace_a, const char* name)
    : Operator(model, strategy, type, name, 2, 0, 1), inplace(inplace_a)
{
}

void
BinaryOperator::Configure(Tensor* input0, Tensor* input1, Tensor* output)
{
  // Hack so that we can access the tensors in the tests
  auto vec_ptr = reinterpret_cast<std::vector<Tensor*>*>(model);
  vec_ptr->emplace_back(input0);
  vec_ptr->emplace_back(input1);
  vec_ptr->emplace_back(output);
}

Legion::Domain
BinaryOperator::GetBounds(Realm::Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
BinaryOperator::Load(Realm::Processor processor)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
void
BinaryOperator::initialize(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
void
BinaryOperator::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
void
BinaryOperator::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
void
BinaryOperator::Free(Realm::Processor processor)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
BinaryOperator::forward_cpu(
    const Legion::Task* task,
    const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
    Legion::Runtime* runtime)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
BinaryOperator::PreregisterTaskVariants(void)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

}}}  // namespace triton::backend::legion
