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

#include "model.h"
#include "common.h"
#include "instance.h"
#include "onnx_parser.h"
#include "operator.h"
#include "tensor.h"

using namespace Legion;

namespace triton {
namespace backend {
namespace legion {

TRITONSERVER_Error *LegionModelState::Create(TRITONBACKEND_Model *triton_model,
                                             std::string const &name,
                                             uint64_t version,
                                             LegionTritonRuntime *runtime,
                                             LegionModelState **state) {
  std::unique_ptr<LegionModelState> lstate;
  try {
    lstate.reset(new LegionModelState(triton_model, runtime, name, version));
  } catch (BackendModelException const &ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr,
        TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelException"));
    RETURN_IF_ERROR(ex.err_);
  }

  // Load the model first to obtain the ground truth for processing model config
  RETURN_IF_ERROR(lstate->LoadModel());

  // Auto-complete the configuration if requested...
  bool auto_complete_config = false;
  RETURN_IF_ERROR(TRITONBACKEND_ModelAutoCompleteConfig(triton_model,
                                                        &auto_complete_config));
  if (auto_complete_config) {
    RETURN_IF_ERROR(lstate->AutoCompleteConfig());

    triton::common::TritonJson::WriteBuffer json_buffer;
    lstate->ModelConfig().Write(&json_buffer);

    TRITONSERVER_Message *message;
    RETURN_IF_ERROR(TRITONSERVER_MessageNewFromSerializedJson(
        &message, json_buffer.Base(), json_buffer.Size()));
    RETURN_IF_ERROR(TRITONBACKEND_ModelSetConfig(
        triton_model, 1 /* config_version */, message));
  }
  RETURN_IF_ERROR(lstate->ValidateModelConfig());
  *state = lstate.release();
  runtime->RecordModel(*state);
  return nullptr; // success
}

LegionModelState::~LegionModelState(void) {
  FreeLayers();
  for (auto &input : inputs_) {
    delete input.second;
  }
  if (strategy_) {
    delete strategy_;
  }
  runtime_->RemoveModel(this);
}

TRITONSERVER_Error *LegionModelState::LoadModel() {
  // TODO: load files based on the default / cc file name that may be set
  // in model config
  auto model_path = JoinPath({RepositoryPath(), std::to_string(Version())});
  assert(strategy_ == nullptr);
  strategy_ = PartitionStrategy::LoadStrategy(
      JoinPath({model_path, "model.strategy"}), this);

  // load the ONNX model description as a list of layers
  // with tensor dependences between then and put them in layers_
  RETURN_IF_ERROR(OnnxParser::LoadModel(
      [this](Realm::Processor::Kind kind)
          -> std::vector<Realm::Processor> const & {
        return runtime_->FindLocalProcessors(kind);
      },
      this,
      strategy_,
      JoinPath({model_path, "model.onnx"}),
      &inputs_,
      &outputs_,
      &layers_));
  RETURN_IF_ERROR(SetOutputInfos());

  // Should have the same number of layers in both cases
  assert(strategy_->layers.size() == layers_.size());

  // Perform the layer fusion optimization based on the partitioning strategy
  FuseLayers();

  // Load each of the layers across the target processors
  LoadLayers();

  return nullptr;
}

unsigned LegionModelState::ReserveInstance(void) {
  AutoLock<true> lock(lock_);
  unsigned result = instances_.size();
  instances_.resize(result + 1, nullptr);
  return result;
}

void LegionModelState::RecordInstance(LegionModelInstance *instance) {
  assert(instance->model_state_ == this);
  AutoLock<true> lock(lock_, false /*exclusive*/);
  assert(instance->index_ < instances_.size());
  assert(instances_[instance->index_] == nullptr);
  instances_[instance->index_] = instance;
}

void LegionModelState::initialize(LegionModelInstance *instance,
                                  unsigned const instance_index,
                                  Runtime *runtime,
                                  Context ctx,
                                  MapperID mapper) {
  // First create logical regions for all the input tensors
  for (auto &input : inputs_) {
    instance->create_tensor_region(input.second);
  }

  for (auto layer : layers_) {
    layer->initialize(instance, instance_index, runtime, ctx, mapper);
  }
}

void LegionModelState::forward(LegionModelInstance *instance,
                               unsigned const instance_index,
                               Runtime *runtime,
                               Context ctx,
                               MapperID mapper,
                               std::vector<InputTensor> const &inputs,
                               std::vector<OutputTensor> const &outputs,
                               std::vector<uint64_t> &compute_input_end_ns,
                               std::vector<uint64_t> &compute_output_start_ns) {
  assert(inputs.size() == inputs_.size());
  assert(outputs.size() == outputs_.size());
  // Attach the external memory allocations to the logical regions for the
  // tensors
  const std::vector<FieldID> fields(1, FID_DATA);
  std::vector<PhysicalRegion> input_regions(inputs.size());
  for (unsigned idx = 0; idx < inputs.size(); idx++) {
    InputTensor const &input = inputs[idx];
    assert(input.buffers_.size() == 1);
    assert(input.buffer_locations_.size() == 1);
    assert(input.buffer_memories_.size() == 1);
    assert(input.strides_.size() == inputs_[idx].second->bounds.size());
    LogicalRegion region = inputs_[idx].second->region[instance_index];
    AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE,
                            region,
                            region,
                            false /*restricted*/,
                            false /*mapped*/);
    launcher.attach_array_soa(const_cast<void *>(input.buffers_[0]),
                              false /*not column major*/,
                              fields,
                              input.buffer_memories_[0]);
    input_regions[idx] = runtime->attach_external_resource(ctx, launcher);
  }
  std::vector<PhysicalRegion> output_regions(outputs.size());
  for (unsigned idx = 0; idx < outputs.size(); idx++) {
    OutputTensor const &output = outputs[idx];
    assert(output.buffers_.size() == 1);
    assert(output.buffer_locations_.size() == 1);
    assert(output.buffer_memories_.size() == 1);
    assert(output.strides_.size() == outputs_[idx].second->bounds.size());
    LogicalRegion region = outputs_[idx].second->region[instance_index];
    AttachLauncher launcher(LEGION_EXTERNAL_INSTANCE,
                            region,
                            region,
                            false /*restricted*/,
                            false /*mapped*/);
    launcher.attach_array_soa(output.buffers_[0],
                              false /*not column major*/,
                              fields,
                              output.buffer_memories_[0]);
    output_regions[idx] = runtime->attach_external_resource(ctx, launcher);
  }
  // Execution fence for timing operation
  runtime->issue_execution_fence(ctx);
  TimingLauncher timing_launcher(LEGION_MEASURE_NANO_SECONDS);
  Future start = runtime->issue_timing_measurement(ctx, timing_launcher);

  // We can trace the execution of this model since it should be the same
  runtime->begin_trace(ctx, 0 /*only ever have one trace*/);
  for (auto layer : layers_) {
    layer->forward(instance, instance_index, runtime, ctx, mapper);
  }
  runtime->end_trace(ctx, 0 /*only ever have one trace*/);

  // Execution fence for timing operation
  runtime->issue_execution_fence(ctx);
  Future stop = runtime->issue_timing_measurement(ctx, timing_launcher);
  // Detach the external memory allocations
  for (unsigned idx = 0; idx < input_regions.size(); idx++) {
    runtime->detach_external_resource(ctx, input_regions[idx], false /*flush*/);
  }
  for (unsigned idx = 0; idx < output_regions.size(); idx++) {
    runtime->detach_external_resource(ctx, output_regions[idx], true /*flush*/);
  }

  const uint64_t start_time = start.get_result<long long>();
  for (unsigned idx = 0; idx < compute_input_end_ns.size(); idx++) {
    compute_input_end_ns[idx] = start_time;
  }

  const uint64_t stop_time = stop.get_result<long long>();
  for (unsigned idx = 0; idx < compute_output_start_ns.size(); idx++) {
    compute_output_start_ns[idx] = stop_time;
  }

  // Wait for everything to be done before we return
  Future done = runtime->issue_execution_fence(ctx);
  done.wait();
}

void LegionModelState::finalize(LegionModelInstance *instance,
                                unsigned const instance_index,
                                Runtime *runtime,
                                Context ctx,
                                MapperID mapper) {
  for (auto layer : layers_) {
    layer->finalize(instance, instance_index, runtime, ctx, mapper);
  }
}

LegionModelInstance *LegionModelState::FindInstance(unsigned instance_index,
                                                    bool external,
                                                    bool need_lock) {
  if (need_lock) {
    if (external) {
      AutoLock<true> lock(lock_, false /*exclusive*/);
      return FindInstance(instance_index, true, false);
    } else {
      AutoLock<false> lock(lock_, false /*exclusive*/);
      return FindInstance(instance_index, false, false);
    }
  }
  assert(instance_index < instances_.size());
  return instances_[instance_index];
}

PartitionStrategy const *LegionModelState::GetStrategy(void) const {
  assert(strategy_ != nullptr);
  return strategy_;
}

TRITONSERVER_Error *LegionModelState::AutoCompleteConfig() {
  // FIXME: Check with the FFModel
  return nullptr; // success
}

TRITONSERVER_Error *LegionModelState::ValidateModelConfig() {
  // Constraints that apply to models in general
  {
    triton::common::TritonJson::Value igs;
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("instance_group", &igs));
    for (size_t i = 0; i < igs.ArraySize(); i++) {
      triton::common::TritonJson::Value ig;
      RETURN_IF_ERROR(igs.IndexAsObject(i, &ig));
      std::string kind_str;
      RETURN_IF_ERROR(ig.MemberAsString("kind", &kind_str));
      if (kind_str != "KIND_MODEL") {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string(
                 "unexpected instance group kind '" + kind_str +
                 "' for model '" + Name() +
                 "', expecting 'KIND_MODEL' to use model specified device "
                 "placement")
                 .c_str()));
      }
    }

    // [issue #4] currently not support batching
    if (max_batch_size_ != 0) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("'max_batch_size' must be 0 in model configuration as "
                       "batching is not currently supported")
               .c_str()));
    }

    // FIXME add check for other model config fields that not yet supported
  }

  {
    // Build a map from name to tensors of the model for easy lookup
    std::map<std::string, Tensor *> tensors;
    for (auto const &io : inputs_) {
      tensors.emplace(io.first, io.second);
    }

    triton::common::TritonJson::Value ios;
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("input", &ios));

    if (ios.ArraySize() != tensors.size()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("configuration for model '" + Name() + "' specifies " +
                       std::to_string(ios.ArraySize()) +
                       " inputs, the model has " +
                       std::to_string(tensors.size()))
               .c_str()));
    }

    for (size_t i = 0; i < ios.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
      std::string io_name;
      RETURN_IF_ERROR(io.MemberAsString("name", &io_name));

      // Check datatypes
      std::string io_dtype;
      RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
      RETURN_ERROR_IF_TRUE((io_dtype == "TYPE_STRING"),
                           TRITONSERVER_ERROR_INVALID_ARG,
                           std::string("unsupported datatype '") + io_dtype +
                               "' for tensor '" + io_name + "' for model '" +
                               Name() + "'");
      // If a reshape is provided for the input then use that when
      // validating that the model matches what is expected.
      std::vector<int64_t> dims;
      triton::common::TritonJson::Value reshape;
      if (io.Find("reshape", &reshape)) {
        RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
      } else {
        RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
      }
      for (auto const dim : dims) {
        RETURN_ERROR_IF_TRUE(
            (dim == WILDCARD_DIM),
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string("dynamic tensor is not supported for model '" + Name() +
                        "'"));
      }

      // Check the properties against the corresponding tensor
      auto it = tensors.find(io_name);
      if (it == tensors.end()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("configuration for model '" + Name() +
                         "' specifies tensor '" + io_name +
                         "' which is not found in the model")
                 .c_str()));
      }
      auto const &tensor = it->second;
      if (ToDataType(ModelConfigDataTypeToTritonServerDataType(io_dtype)) !=
          tensor->type) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("configuration for model '" + Name() +
                         "' specifies tensor '" + io_name + "' with type '" +
                         io_dtype + "', the tensor in the model has type '" +
                         DataTypeString(tensor->type) + "'")
                 .c_str()));
      } else if (tensor->type == DT_NONE) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("tensor '" + io_name + "' in the model '" + Name() +
                         "' has unknown type")
                 .c_str()));
      }
      if (max_batch_size_ != 0) {
        dims.insert(dims.begin(), max_batch_size_);
      }
      // put tensor's bound in int64_t to utilize backend common utilities
      std::vector<int64_t> tensor_bounds;
      for (auto const bound : tensor->bounds) {
        tensor_bounds.emplace_back(bound);
      }
      if (dims != tensor_bounds) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("configuration for model '" + Name() +
                         "' specifies tensor '" + io_name +
                         "' with full shape " + ShapeToString(dims) +
                         ", the tensor in the model has shape " +
                         ShapeToString(tensor_bounds))
                 .c_str()));
      }
    }
  }

  // Outputs
  {
    // Build a map from name to tensors of the model for easy lookup
    std::map<std::string, Tensor *> tensors;
    for (auto const &io : outputs_) {
      tensors.emplace(io.first, io.second);
    }

    triton::common::TritonJson::Value ios;
    RETURN_IF_ERROR(ModelConfig().MemberAsArray("output", &ios));

    // Model config may expose a subset of the outputs
    if (ios.ArraySize() > tensors.size()) {
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_INVALID_ARG,
          (std::string("configuration for model '" + Name() + "' specifies " +
                       std::to_string(ios.ArraySize()) +
                       " outputs, the model has " +
                       std::to_string(tensors.size()))
               .c_str()));
    }

    for (size_t i = 0; i < ios.ArraySize(); i++) {
      triton::common::TritonJson::Value io;
      RETURN_IF_ERROR(ios.IndexAsObject(i, &io));
      std::string io_name;
      RETURN_IF_ERROR(io.MemberAsString("name", &io_name));
      // Check datatypes
      std::string io_dtype;
      RETURN_IF_ERROR(io.MemberAsString("data_type", &io_dtype));
      RETURN_ERROR_IF_TRUE((io_dtype == "TYPE_STRING"),
                           TRITONSERVER_ERROR_INVALID_ARG,
                           std::string("unsupported datatype '") + io_dtype +
                               "' for tensor '" + io_name + "' for model '" +
                               Name() + "'");
      // If a reshape is provided for the input then use that when
      // validating that the model matches what is expected.
      std::vector<int64_t> dims;
      triton::common::TritonJson::Value reshape;
      if (io.Find("reshape", &reshape)) {
        RETURN_IF_ERROR(ParseShape(reshape, "shape", &dims));
      } else {
        RETURN_IF_ERROR(ParseShape(io, "dims", &dims));
      }
      for (auto const dim : dims) {
        RETURN_ERROR_IF_TRUE(
            (dim == WILDCARD_DIM),
            TRITONSERVER_ERROR_INVALID_ARG,
            std::string("dynamic tensor is not supported for model '" + Name() +
                        "'"));
      }

      // Check the properties against the corresponding tensor
      auto it = tensors.find(io_name);
      if (it == tensors.end()) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("configuration for model '" + Name() +
                         "' specifies tensor '" + io_name +
                         "' which is not found in the model")
                 .c_str()));
      }
      auto const &tensor = it->second;
      if (ToDataType(ModelConfigDataTypeToTritonServerDataType(io_dtype)) !=
          tensor->type) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("configuration for model '" + Name() +
                         "' specifies tensor '" + io_name + "' with type '" +
                         io_dtype + "', the tensor in the model has type '" +
                         DataTypeString(tensor->type) + "'")
                 .c_str()));
      } else if (tensor->type == DT_NONE) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("tensor '" + io_name + "' in the model '" + Name() +
                         "' has unknown type")
                 .c_str()));
      }
      if (max_batch_size_ != 0) {
        dims.insert(dims.begin(), max_batch_size_);
      }
      // put tensor's bound in int64_t to utilize backend common utilities
      std::vector<int64_t> tensor_bounds;
      for (auto const bound : tensor->bounds) {
        tensor_bounds.emplace_back(bound);
      }
      if (dims != tensor_bounds) {
        return TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INVALID_ARG,
            (std::string("configuration for model '" + Name() +
                         "' specifies tensor '" + io_name +
                         "' with full shape " + ShapeToString(dims) +
                         ", the tensor in the model has shape " +
                         ShapeToString(tensor_bounds))
                 .c_str()));
      }
    }
  }
  return nullptr; // success
}

TRITONSERVER_Error *LegionModelState::SetOutputInfos() {
  for (auto const &output : outputs_) {
    std::vector<int64_t> tensor_bounds;
    for (auto const bound : output.second->bounds) {
      tensor_bounds.emplace_back(bound);
    }
    auto triton_dtype = ToTritonDataType(output.second->type);
    output_infos_.emplace_back(output.first, triton_dtype, tensor_bounds);
  }
  return nullptr; // success
}

void LegionModelState::LoadLayers(void) const {
  std::vector<Realm::Event> loaded_events;
  for (unsigned idx1 = 0; idx1 < layers_.size(); idx1++) {
    Operator *op = layers_[idx1];
    LayerStrategy const *config = strategy_->layers[idx1];
    for (unsigned idx2 = 0; idx2 < config->nProcs; idx2++) {
      Realm::Processor proc = config->local_processors[idx2];
      loaded_events.push_back(runtime_->LoadLayer(proc, op));
    }
  }
  const Realm::Event wait_on = Realm::Event::merge_events(loaded_events);
  if (wait_on.exists() && !wait_on.has_triggered()) {
    wait_on.external_wait();
  }
}

void LegionModelState::FuseLayers(void) {
  // FIXME: add support for layer fusion
}

void LegionModelState::FreeLayers(void) const {
  std::vector<Realm::Event> freed_events;
  for (unsigned idx1 = 0; idx1 < layers_.size(); idx1++) {
    Operator *op = layers_[idx1];
    LayerStrategy const *config = strategy_->layers[idx1];
    for (unsigned idx2 = 0; idx2 < config->nProcs; idx2++) {
      Realm::Processor proc = config->local_processors[idx2];
      freed_events.push_back(runtime_->FreeLayer(proc, op));
    }
  }
  const Realm::Event wait_on = Realm::Event::merge_events(freed_events);
  if (wait_on.exists() && !wait_on.has_triggered()) {
    wait_on.external_wait();
  }
  // Delete layers back to front
  for (std::vector<Operator *>::const_reverse_iterator it = layers_.rbegin();
       it != layers_.rend();
       it++) {
    delete (*it);
  }
}

} // namespace legion
} // namespace backend
} // namespace triton
