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

#ifndef __LEGION_TRITON_CONCAT_H__
#define __LEGION_TRITON_CONCAT_H__

#include "operator.h"
#include "tensor.h"
#ifdef LEGION_USE_CUDA
#include "cudahelp.h"
#endif

namespace triton {
namespace backend {
namespace legion {

class FilterProjectionFunctor : public Legion::ProjectionFunctor {
public:
  virtual Legion::LogicalRegion project(Legion::LogicalPartition upper_bound,
                                        Legion::DomainPoint const &point,
                                        Legion::Domain const &domain) override;

public:
  virtual bool is_functional(void) const override {
    return true;
  }
  virtual unsigned get_depth(void) const override {
    return 0;
  }
};

struct ConcatArgs : public OperatorArgs {
public:
  ConcatArgs(void);

public:
  unsigned local_index;
  Legion::Domain bounds;
  DataType datatype;
  int axis;
};

class Concat : public Operator {
public:
  Concat(LegionModelState *model,
         LayerStrategy const *strategy,
         size_t inputs,
         int axis,
         char const *name);

public:
  void Configure(std::vector<Tensor *> const &inputs, Tensor *output);
  Legion::Domain GetBounds(Realm::Processor proc);

public:
  virtual void Load(Realm::Processor processor) override;
  virtual void initialize(LegionModelInstance *instance,
                          unsigned const instance_index,
                          Legion::Runtime *runtime,
                          Legion::Context ctx,
                          Legion::MapperID mapper) override;
  virtual void forward(LegionModelInstance *instance,
                       unsigned const instance_index,
                       Legion::Runtime *runtime,
                       Legion::Context ctx,
                       Legion::MapperID mapper) override;
  virtual void finalize(LegionModelInstance *instance,
                        unsigned const instance_index,
                        Legion::Runtime *runtime,
                        Legion::Context ctx,
                        Legion::MapperID mapper) override;
  virtual void Free(Realm::Processor processor) override;

public:
  static void PreregisterTaskVariants(void);
  static void forward_cpu(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);
#ifdef LEGION_USE_CUDA
public:
  // Forward task for the GPU
  static void forward_gpu(Legion::Task const *task,
                          std::vector<Legion::PhysicalRegion> const &regions,
                          Legion::Context ctx,
                          Legion::Runtime *runtime);
#endif
public:
  int const axis;
  static Legion::ProjectionID filter_functor_id;

protected:
  ConcatArgs args[MAX_LOCAL_PROCS];
  Legion::FutureMap argmaps[MAX_NUM_INSTANCES];
  Legion::IndexTaskLauncher launchers[MAX_NUM_INSTANCES];
  std::vector<Legion::Domain> input_color_spaces;
  std::vector<Legion::Domain> input_extents;
  Legion::DomainTransform input_transform;
};

} // namespace legion
} // namespace backend
} // namespace triton

#endif // __LEGION_TRITON_CONCAT_H__
