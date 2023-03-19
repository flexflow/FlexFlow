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

#ifndef __LEGION_TRITON_OPERATOR_H__
#define __LEGION_TRITON_OPERATOR_H__

#include "accessor.h"
#include "config.h"
#include "instance.h"
#include "legion.h"
#include "model.h"
#include "strategy.h"
#include "types.h"

namespace triton {
namespace backend {
namespace legion {

class Operator;

struct OperatorArgs {
public:
  OperatorArgs(bool prof = false) : profiling(prof), owner(nullptr) {}

public:
  bool profiling;
  Operator *owner; // technically not legion safe, debugging/profiling only
#if 0
  cudnnHandle_t dnn;
  cublasHandle_t blas;
  bool allowTensorOpMathConversion;
#ifdef USE_NCCL
  ncclComm_t ncclComm;
#endif
#endif
};

class Operator {
public:
  Operator(LegionModelState *model,
           LayerStrategy const *strategy,
           OperatorType type,
           char const *name,
           unsigned num_inputs,
           unsigned num_weights,
           unsigned num_outputs);
  virtual ~Operator(void);

public:
  // Called by model load (Realm)
  virtual void Load(Realm::Processor processor) = 0;
  // Called per instance (Legion)
  virtual void initialize(LegionModelInstance *instance,
                          unsigned const instance_index,
                          Legion::Runtime *runtime,
                          Legion::Context ctx,
                          Legion::MapperID mapper) = 0;
  virtual void forward(LegionModelInstance *instance,
                       unsigned const instance_index,
                       Legion::Runtime *runtime,
                       Legion::Context ctx,
                       Legion::MapperID mapper) = 0;
  virtual void finalize(LegionModelInstance *instance,
                        unsigned const instance_index,
                        Legion::Runtime *runtime,
                        Legion::Context ctx,
                        Legion::MapperID mapper) = 0;
  // Called by model free (Realm)
  virtual void Free(Realm::Processor processor) = 0;

public:
  static void PreregisterTaskVariants(void);

public:
  const OperatorType op_type;
  const std::string op_name;
  LegionModelState *const model;
  LayerStrategy const *const strategy;
  unsigned const num_inputs;
  unsigned const num_weights;
  unsigned const num_outputs;

protected:
  Legion::IndexSpace launch_space[MAX_NUM_INSTANCES];
  std::vector<Tensor *> inputs;
  std::vector<Tensor *> outputs;
  std::vector<Weights *> weights;
};

} // namespace legion
} // namespace backend
} // namespace triton

#endif // __LEGION_TRITON_OPERATOR_H__
