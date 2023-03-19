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

#ifndef __LEGION_TRITON_INSTANCE_H__
#define __LEGION_TRITON_INSTANCE_H__

#include "legion.h"
#include "model.h"
#include "runtime.h"
#include "strategy.h"
#include "triton/backend/backend_input_collector.h"
#include "triton/backend/backend_memory.h"
#include "triton/backend/backend_model_instance.h"

namespace triton { namespace backend { namespace legion {

struct InputTensor {
  std::string name_;
  std::vector<const void*> buffers_;
  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> buffer_locations_;
  std::vector<Realm::Memory> buffer_memories_;
  std::vector<int64_t> strides_;
  // A placeholder for the memory acquired to hold the preprocessed input buffer
  std::vector<std::unique_ptr<BackendMemory>> allocated_memory_;
};

struct OutputTensor {
  std::string name_;
  std::vector<void*> buffers_;
  std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> buffer_locations_;
  std::vector<Realm::Memory> buffer_memories_;
  std::vector<int64_t> strides_;
  // A placeholder for the memory acquired to hold the part of the output
  // that is not requested
  std::vector<std::unique_ptr<BackendMemory>> allocated_memory_;
};

//
// LegionModelInstance
//
// Each instantiation of this class represents a backend instance
// for running inference requests on a set of resources. It will
// have an associated Legion implicit task along with a trace for
// replaying inference jobs.
//
//
class LegionModelInstance : public BackendModelInstance {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_ModelInstance* triton_model_instance,
      LegionModelState* model_state, LegionModelInstance** state);

  ~LegionModelInstance();

  void CreateContext(
      Legion::Runtime* runtime, Legion::TaskID tid, unsigned rank,
      size_t total_ranks, Realm::Event precondition, bool owner_instance);

  Realm::Barrier GetExecutionBarrier(
      size_t ranks, Realm::Event& precondition, bool external,
      bool need_lock = true);

  // Execute...
  void ProcessRequests(
      TRITONBACKEND_Request** requests, const uint32_t request_count);

  void RunModel(
      const std::vector<InputTensor>& inputs,
      const std::vector<OutputTensor>& outputs,
      std::vector<uint64_t>& compute_input_end,
      std::vector<uint64_t>& compute_output_start, bool distributed = false);

 private:
  inline void Bind() const
  {
    runtime_->bind_implicit_task_to_external_thread(context_);
  }
  inline void Unbind() const
  {
    runtime_->unbind_implicit_task_from_external_thread(context_);
  }

  // Small helper class to make sure we always unbind even under errors
  class AutoBind {
   public:
    AutoBind(LegionModelInstance* state) : instance_state(state)
    {
      state->Bind();
    }
    ~AutoBind() { instance_state->Unbind(); }

   private:
    LegionModelInstance* const instance_state;
  };

  // Set the input tensors for running the model, in case of error, responses
  // will be returned with error and the function will return false.
  // Returns true on success.
  bool SetInputTensors(
      const size_t total_batch_size, TRITONBACKEND_Request** requests,
      const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      std::vector<InputTensor>& inputs);

  bool SetOutputTensors(
      const size_t total_batch_size,
      const std::vector<size_t>& request_batch_sizes,
      TRITONBACKEND_Request** requests, const uint32_t request_count,
      std::vector<TRITONBACKEND_Response*>* responses,
      std::vector<OutputTensor>& outputs);

  LegionModelInstance(
      TRITONBACKEND_ModelInstance* triton_model_instance,
      LegionModelState* model_state, unsigned index, Realm::Event ready);

 public:
  // methods for support of operators
  // should only be invoked inside the implicit top-level legion tasks
  Legion::IndexSpace find_or_create_index_space(const Legion::Domain& domain);
  Legion::IndexPartition find_or_create_partition(
      Legion::IndexSpace top_level_space, Legion::IndexSpace color_space,
      const Legion::DomainTransform& transform, const Legion::Domain& extent,
      Legion::PartitionKind kind);
  Legion::FieldSpace find_or_create_field_space(DataType date_type);
  Legion::LogicalRegion create_tensor_region(Tensor* tensor);
  Legion::LogicalPartition find_or_create_tiled_partition(
      Tensor* tensor, const LayerStrategy* strategy);

 public:
  Legion::Runtime* const runtime_;
  LegionModelState* const model_state_;
  const unsigned index_;
  const Realm::Event context_ready_;

 private:
  Legion::Context context_;
  Legion::MapperID mapper_;

 private:
  Realm::FastReservation lock_;
  Realm::Barrier execution_barrier_;

 private:
  std::map<Legion::Domain, Legion::IndexSpace> top_level_index_spaces;
  struct Partition {
   public:
    Partition(void) {}
    Partition(
        Legion::IndexSpace cs, Legion::IndexPartition p,
        const Legion::DomainTransform& t, const Legion::Domain& e)
        : color_space(cs), partition(p), transform(t), extent(e)
    {
    }

   public:
    Legion::IndexSpace color_space;
    Legion::IndexPartition partition;
    Legion::DomainTransform transform;
    Legion::Domain extent;
  };
  std::map<Legion::IndexSpace, std::vector<Partition>> top_level_partitions;
  std::map<DataType, Legion::FieldSpace> top_level_field_spaces;
  std::vector<Legion::LogicalRegion> top_level_regions;
};

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_INSTANCE_H__
