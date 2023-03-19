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

#ifndef __LEGION_TRITON_MATMUL_H__
#define __LEGION_TRITON_MATMUL_H__

#include "operator.h"
#include "tensor.h"

namespace triton { namespace backend { namespace legion {

class MatMulProjectionFunctor : public Legion::ProjectionFunctor {
 public:
  MatMulProjectionFunctor(
      Legion::ProjectionID id, const Legion::DomainTransform& transform);

 public:
  inline Legion::DomainPoint transform(const Legion::DomainPoint& point) const
  {
    return domain_transform * point;
  }

 public:
  virtual bool is_functional(void) const override { return true; }
  virtual unsigned get_depth(void) const override { return 0; }
  virtual Legion::LogicalRegion project(
      Legion::LogicalPartition upper_bound, const Legion::DomainPoint& point,
      const Legion::Domain& domain) override;

 public:
  const Legion::ProjectionID functor_id;
  const Legion::DomainTransform domain_transform;
};

class MatMul;

struct MatMulArgs : public OperatorArgs {
 public:
  MatMulArgs(void);
  MatMul* owner;
  Legion::Domain in1_bounds, in2_bounds, out_bounds;
  DataType in1_datatype, in2_datatype, out_datatype;
#ifdef LEGION_USE_CUDA
  cublasHandle_t cublas;
#endif
};

class MatMul : public Operator {
 public:
  MatMul(
      LegionModelState* model, const LayerStrategy* strategy, const char* name);

  void Configure(Tensor* in1, Tensor* in2, Tensor* output);
  Legion::Domain GetIn1Bounds(Realm::Processor proc);
  Legion::Domain GetIn2Bounds(Realm::Processor proc);
  Legion::Domain GetOutBounds(Realm::Processor proc);

  virtual void Load(Realm::Processor processor) override;
  virtual void initialize(
      LegionModelInstance* instance, const unsigned instance_index,
      Legion::Runtime* runtime, Legion::Context ctx,
      Legion::MapperID mapper) override;
  virtual void forward(
      LegionModelInstance* instance, const unsigned instance_index,
      Legion::Runtime* runtime, Legion::Context ctx,
      Legion::MapperID mapper) override;
  virtual void finalize(
      LegionModelInstance* instance, const unsigned instance_index,
      Legion::Runtime* runtime, Legion::Context ctx,
      Legion::MapperID mapper) override;
  virtual void Free(Realm::Processor processor) override;

  static void PreregisterTaskVariants(void);

  static void forward_cpu(
      const Legion::Task* task,
      const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
      Legion::Runtime* runtime);

#ifdef LEGION_USE_CUDA
  static void forward_gpu(
      const Legion::Task* task,
      const std::vector<Legion::PhysicalRegion>& regions, Legion::Context ctx,
      Legion::Runtime* runtime);
#endif
 protected:
  template <unsigned DIM>
  void compute_in1_parameters(Tensor* in1, Tensor* out);
  template <unsigned DIM>
  void compute_in2_parameters(Tensor* in2, Tensor* out);

 protected:
  template <unsigned DIM>
  static void generate_all_functors(void);
  template <unsigned IDIM, unsigned ODIM>
  static void generate_specific_functors(void);

 protected:
  MatMulArgs args[MAX_LOCAL_PROCS];
  MatMulProjectionFunctor *in1_proj, *in2_proj;
  Legion::DomainTransform in1_transform, in2_transform;
  Legion::Domain in1_extent, in2_extent;
  Legion::Domain in1_colors, in2_colors;
  Legion::FutureMap argmaps[MAX_NUM_INSTANCES];
  Legion::IndexTaskLauncher launchers[MAX_NUM_INSTANCES];

 public:
  // for looking up projection functor IDs
  // keys are <input dimensions,output dimensions,one-hot encoded broadcasting>
  typedef std::tuple<unsigned, unsigned, unsigned> FunctorKey;
  typedef std::map<FunctorKey, MatMulProjectionFunctor*> FunctorTable;
  static FunctorTable in1_functors, in2_functors;
};

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_MATMUL_H__
