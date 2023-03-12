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

#include "operators/concat.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

ConcatArgs::ConcatArgs(void) : local_index(0), datatype(DT_NONE), axis(-1) {}

Concat::Concat(
    LegionModelState* model, const LayerStrategy* strategy, size_t inputs,
    int ax, const char* name)
    : Operator(model, strategy, OperatorType::OP_CONCAT, name, inputs, 0, 1),
      axis(ax)
{
  assert(inputs > 0);
}

void
Concat::Configure(const std::vector<Tensor*>& ins, Tensor* out)
{
  assert(num_inputs == ins.size());
  inputs = ins;
  size_t axis_size = 0;
  const size_t dims = out->bounds.size();
  assert(dims == strategy->nDims);
  for (unsigned idx = 0; idx < inputs.size(); idx++) {
    assert(inputs[idx]->type == out->type);
    assert(inputs[idx]->bounds.size() == dims);
    for (unsigned d = 0; d < dims; d++) {
      if (d == axis)
        axis_size += inputs[idx]->bounds[d];
      else
        assert(inputs[idx]->bounds[d] == out->bounds[d]);
    }
  }
  assert(axis_size == out->bounds[axis]);
  outputs.push_back(out);
  // Figure out the output tiling domain
  std::vector<size_t> tile_sizes(dims);
  for (unsigned d = 0; d < dims; d++)
    tile_sizes[d] = (out->bounds[d] + strategy->dim[d] - 1) / strategy->dim[d];
  coord_t offset = 0;
  // Now compute the domains and transforms needed for constructing
  // the partitions for each of the inputs
  input_color_spaces.resize(num_inputs);
  input_extents.resize(num_inputs);
  for (unsigned idx = 0; idx < num_inputs; idx++) {
    DomainPoint lo, hi, color_lo, color_hi;
    lo.dim = dims;
    hi.dim = dims;
    color_lo.dim = dims;
    color_hi.dim = dims;
    for (int d = 0; d < dims; d++) {
      if (d == axis) {
        const coord_t extent = inputs[idx]->bounds[d];
        lo[d] = -offset;
        hi[d] = (tile_sizes[d] - 1 /*inclusive*/) - offset;
        color_lo[d] = offset / tile_sizes[d];
        color_hi[d] = (offset + extent - 1) / tile_sizes[d];
        offset += extent;
      } else {
        lo[d] = 0;
        hi[d] = tile_sizes[d] - 1;  // make it inclusive
        color_lo[d] = 0;
        color_hi[d] = strategy->dim[d] - 1;  // make it inclusive
      }
    }
    input_color_spaces[idx] = Domain(color_lo, color_hi);
    input_extents[idx] = Domain(lo, hi);
  }
  // The input transform is the same across all the inputs
  switch (dims) {
#define DIMFUNC(N)                         \
  case N: {                                \
    Transform<N, N> transform;             \
    for (int i = 0; i < N; i++)            \
      for (int j = 0; j < N; j++)          \
        if (i == j)                        \
          transform[i][j] = tile_sizes[i]; \
        else                               \
          transform[i][j] = 0;             \
    input_transform = transform;           \
    break;                                 \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      abort();
  }
}

Domain
Concat::GetBounds(Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
Concat::Load(Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
Concat::initialize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
Concat::forward(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
Concat::finalize(
    LegionModelInstance* instance, const unsigned instance_index,
    Runtime* runtime, Context ctx, MapperID mapper)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

void
Concat::Free(Processor proc)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

/*static*/ void
Concat::PreregisterTaskVariants(void)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

/*static*/ void
Concat::forward_cpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}

#ifdef LEGION_USE_CUDA
/*static*/ void
Concat::forward_gpu(
    const Task* task, const std::vector<PhysicalRegion>& regions, Context ctx,
    Runtime* runtime)
{
  throw std::invalid_argument(
      "This function shouldn't be called in parser unit test");
}
#endif

}}}  // namespace triton::backend::legion
