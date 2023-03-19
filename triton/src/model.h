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

#ifndef __LEGION_TRITON_MODEL_H__
#define __LEGION_TRITON_MODEL_H__

#include "legion.h"
#include "triton/backend/backend_common.h"
#include "triton/backend/backend_model.h"
#include "types.h"

namespace triton { namespace backend { namespace legion {

//
// LegionModelState
//
// Capture the meta data needed for representing a model
//
class LegionModelState : public BackendModel {
 public:
  static TRITONSERVER_Error* Create(
      TRITONBACKEND_Model* triton_model, const std::string& name,
      uint64_t version, LegionTritonRuntime* runtime, LegionModelState** state);
  virtual ~LegionModelState();


  unsigned ReserveInstance(void);
  void RecordInstance(LegionModelInstance* instance);

  LegionModelInstance* FindInstance(
      unsigned instance_index, bool external, bool need_lock = true);
  const PartitionStrategy* GetStrategy(void) const;

  // These methods must all be called while the instance is bound
  // to the its implicit top-level task context
  void initialize(
      LegionModelInstance* instance, const unsigned instance_index,
      Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper);
  void forward(
      LegionModelInstance* instance, const unsigned instance_index,
      Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper,
      const std::vector<InputTensor>& inputs,
      const std::vector<OutputTensor>& outputs,
      std::vector<uint64_t>& compute_input_end_ns,
      std::vector<uint64_t>& compute_output_end_ns);
  void finalize(
      LegionModelInstance* instance, const unsigned instance_index,
      Legion::Runtime* runtime, Legion::Context ctx, Legion::MapperID mapper);
  const std::vector<
      std::tuple<std::string, TRITONSERVER_DataType, std::vector<int64_t>>>&
  OutputInfos()
  {
    return output_infos_;
  }

 private:
  LegionModelState(
      TRITONBACKEND_Model* triton_model, LegionTritonRuntime* runtime,
      const std::string& n, uint64_t v)
      : BackendModel(triton_model), runtime_(runtime), name(n), version(v),
        strategy_(nullptr)
  {
  }

  TRITONSERVER_Error* LoadModel();
  TRITONSERVER_Error* AutoCompleteConfig();
  TRITONSERVER_Error* ValidateModelConfig();
  TRITONSERVER_Error* SetOutputInfos();

  void LoadLayers(void) const;
  void FuseLayers(void);
  void FreeLayers(void) const;

 public:
  LegionTritonRuntime* const runtime_;
  const std::string name;
  const uint64_t version;

 private:
  Realm::FastReservation lock_;
  std::vector<std::pair<std::string, Tensor*>> inputs_;  // We own these tensors
  std::vector<std::pair<std::string, Tensor*>>
      outputs_;  // We do NOT own these tensors
  std::vector<Operator*> layers_;
  PartitionStrategy* strategy_;
  std::vector<LegionModelInstance*> instances_;
  // Output information parsed from 'outputs_' for easier access,
  // use to interact with Triton APIs.
  // FIXME calculate stride once for all
  std::vector<
      std::tuple<std::string, TRITONSERVER_DataType, std::vector<int64_t>>>
      output_infos_;
};

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_MODEL_H__
