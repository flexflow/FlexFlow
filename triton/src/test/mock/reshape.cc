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

#include "operators/reshape.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

Reshape::Reshape(
    LegionModelState* model, const LayerStrategy* strategy, const char* name)
    : Operator(model, strategy, OperatorType::OP_RESHAPE, name, 1, 0, 1)
{
}

void
Reshape::Configure(Tensor* input, Tensor* output)
{
  assert(input != nullptr);
  assert(output != nullptr);
  assert(input->type == output->type);
  // Make sure that they have the same volumes
  size_t input_volume = 1, output_volume = 1;
  for (unsigned idx = 0; idx < input->bounds.size(); idx++)
    input_volume *= input->bounds[idx];
  for (unsigned idx = 0; idx < output->bounds.size(); idx++)
    output_volume *= output->bounds[idx];
  assert(input_volume == output_volume);

  // Group dimensions from the two input tensors together from
  // right-to-left to find ones that can be tiles together
  int input_idx = input->bounds.size() - 1;
  int output_idx = output->bounds.size() - 1;
  while ((input_idx >= 0) && (output_idx >= 0)) {
    std::vector<int> input_dims(1, input_idx);
    std::vector<int> output_dims(1, output_idx);
    size_t input_tile_volume = input->bounds[input_idx--];
    size_t output_tile_volume = output->bounds[output_idx--];
    while (input_tile_volume != output_tile_volume) {
      if (input_tile_volume < output_tile_volume) {
        input_dims.push_back(input_idx);
        input_tile_volume *= input->bounds[input_idx--];
      } else {
        output_dims.push_back(output_idx);
        output_tile_volume *= output->bounds[output_idx--];
      }
    }
    input_groups.emplace_back(input_dims);
    output_groups.emplace_back(output_dims);
  }
  // In order to use the output launch space, we need to make sure that
  // all but the earliest dimension in each output group has a partitioning
  // strategy of 1 or else we won't be able to compute a partition that
  // will allow for densely tiled copies. In the future we could fix this
  // by computing a generalized index launch space and then mapping that
  // onto the original output launch space or just by using affine indirect
  // copy launchers when they are available.
  for (unsigned g = 0; g < output_groups.size(); g++) {
    const std::vector<int>& input_group = input_groups[g];
    const std::vector<int>& output_group = output_groups[g];
    for (unsigned idx = 0; idx < (output_group.size() - 1); idx++)
      assert(strategy->dim[output_group[idx]] == 1);
    // the size of the earliest dimension in the input group must also
    // be divisible by the number of chunks
    assert(
        (input->bounds[input_group.back()] %
         strategy->dim[output_group.back()]) == 0);
    // the output bounds also need to be evenly divisible too or this will not
    // work
    assert(
        (output->bounds[output_group.back()] %
         strategy->dim[output_group.back()]) == 0);
  }
  inputs.push_back(input);
  outputs.push_back(output);

  // Hack so that we can access the tensors in the tests
  auto vec_ptr = reinterpret_cast<std::vector<Tensor*>*>(model);
  vec_ptr->emplace_back(input);
  vec_ptr->emplace_back(output);
}

Domain
Reshape::GetInputBounds(Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

Domain
Reshape::GetOutputBounds(Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
Reshape::Load(Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
Reshape::initialize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
Reshape::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
Reshape::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
Reshape::Free(Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

/*static*/ void
Reshape::PreregisterTaskVariants(void)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

/*static*/ void
Reshape::forward_cpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

ReshapeArgs::ReshapeArgs(void) {}

#ifdef LEGION_USE_CUDA
/*static*/ void
Reshape::forward_gpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
#endif

}}}  // namespace triton::backend::legion
