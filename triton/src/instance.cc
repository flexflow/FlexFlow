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

#include "instance.h"
#include "strategy.h"
#include "tensor.h"

#define RESPOND_ALL_AND_RETURN_IF_ERROR(RET, RESPONSES, RESPONSES_COUNT, X) \
  do {                                                                      \
    TRITONSERVER_Error* raarie_err__ = (X);                                 \
    if (raarie_err__ != nullptr) {                                          \
      SendErrorForResponses(RESPONSES, RESPONSES_COUNT, raarie_err__);      \
      return RET;                                                           \
    }                                                                       \
  } while (false)

#define GUARDED_RESPOND_IF_ERROR(RESPONSES, IDX, X)                     \
  do {                                                                  \
    if ((RESPONSES)[IDX] != nullptr) {                                  \
      TRITONSERVER_Error* err__ = (X);                                  \
      if (err__ != nullptr) {                                           \
        LOG_IF_ERROR(                                                   \
            TRITONBACKEND_ResponseSend(                                 \
                (RESPONSES)[IDX], TRITONSERVER_RESPONSE_COMPLETE_FINAL, \
                err__),                                                 \
            "failed to send error response");                           \
        (RESPONSES)[IDX] = nullptr;                                     \
        TRITONSERVER_ErrorDelete(err__);                                \
      }                                                                 \
    }                                                                   \
  } while (false)

using namespace Legion;

namespace triton { namespace backend { namespace legion {

TRITONSERVER_Error*
LegionModelInstance::Create(
    TRITONBACKEND_ModelInstance* triton_model_instance,
    LegionModelState* model_state, LegionModelInstance** state)
{
  // Make a user event to denote when the context will be ready for this
  // instance
  Realm::UserEvent context_ready = Realm::UserEvent::create_user_event();
  unsigned index = model_state->ReserveInstance();
  try {
    *state = new LegionModelInstance(
        triton_model_instance, model_state, index, context_ready);
  }
  catch (const BackendModelInstanceException& ex) {
    RETURN_ERROR_IF_TRUE(
        ex.err_ == nullptr, TRITONSERVER_ERROR_INTERNAL,
        std::string("unexpected nullptr in BackendModelInstanceException"));
    RETURN_IF_ERROR(ex.err_);
  }
  model_state->RecordInstance(*state);
  model_state->runtime_->RendezvousContextCreation(*state, context_ready);

  return nullptr;  // success
}

LegionModelInstance::~LegionModelInstance()
{
  // Finish the implicit top-level task associated with this instance
  Bind();
  model_state_->finalize(this, index_, runtime_, context_, mapper_);
  for (std::vector<LogicalRegion>::const_iterator it =
           top_level_regions.begin();
       it != top_level_regions.end(); it++)
    runtime_->destroy_logical_region(context_, *it);
  for (std::map<DataType, FieldSpace>::const_iterator it =
           top_level_field_spaces.begin();
       it != top_level_field_spaces.end(); it++)
    runtime_->destroy_field_space(context_, it->second);
  for (std::map<Domain, IndexSpace>::const_iterator it =
           top_level_index_spaces.end();
       it != top_level_index_spaces.end(); it++)
    runtime_->destroy_index_space(context_, it->second);
  // FIXME: find a way to tell Legion to delete our mapper
  runtime_->finish_implicit_task(context_);
}

LegionModelInstance::LegionModelInstance(
    TRITONBACKEND_ModelInstance* triton_model_instance,
    LegionModelState* model_state, unsigned index, Realm::Event ready)
    : BackendModelInstance(model_state, triton_model_instance),
      runtime_(model_state->runtime_->legion_), model_state_(model_state),
      index_(index), context_ready_(ready), mapper_(0)
{
  execution_barrier_ = Realm::Barrier::NO_BARRIER;
}

void
LegionModelInstance::CreateContext(
    Runtime* runtime, TaskID tid, unsigned rank, size_t total_ranks,
    Realm::Event precondition, bool owner_instance)
{
  context_ = runtime->begin_implicit_task(
      tid, 0 /*default mapper to bootstrap only*/, Processor::LOC_PROC /*CPU*/,
      "Inference Task", true /*control replicable*/, 1 /*shard per process*/,
      rank /*shard id*/);
  // Create a unique mapper ID and mapper for this instance and then load the
  // mapper
  assert(mapper_ == 0);
  // this will generate the same ID across the shards
  mapper_ = runtime->generate_dynamic_mapper_id();
  assert(mapper_ != 0);
  StrategyMapper* mapper = new StrategyMapper(
      model_state_->GetStrategy(), runtime->get_mapper_runtime(),
      Machine::get_machine());
  // Register this mapper with all the processors on the local node
  runtime_->add_mapper(mapper_, mapper);

  model_state_->initialize(this, index_, runtime_, context_, mapper_);
  // we can immediately unbind from this context
  Unbind();
  // Check to see if we'll be the owner for managing execution
  assert(!execution_barrier_.exists());
  if (owner_instance) {
    execution_barrier_ = Realm::Barrier::create_barrier(total_ranks);
    // The first generation is just our normal precondition
    execution_barrier_.arrive(total_ranks, precondition);
    execution_barrier_ = execution_barrier_.advance_barrier();
  }
}

Realm::Barrier
LegionModelInstance::GetExecutionBarrier(
    size_t total_ranks, Realm::Event& precondition, bool external,
    bool need_lock)
{
  if (need_lock) {
    if (external) {
      AutoLock<true> lock(lock_);
      return GetExecutionBarrier(total_ranks, precondition, true, false);
    } else {
      AutoLock<false> lock(lock_);
      return GetExecutionBarrier(total_ranks, precondition, false, false);
    }
  }
  // This better exist if we're here
  assert(execution_barrier_.exists());
  precondition = execution_barrier_.get_previous_phase();
  const Realm::Barrier result = execution_barrier_;
  execution_barrier_ = execution_barrier_.advance_barrier();
  if (!execution_barrier_.exists()) {
    // Handle the case where we run out of barrier generations
    execution_barrier_ = Realm::Barrier::create_barrier(total_ranks);
    // Chain the barriers together in order
    execution_barrier_.arrive(total_ranks, result);
    execution_barrier_ = execution_barrier_.advance_barrier();
  }
  return result;
}

void
LegionModelInstance::RunModel(
    const std::vector<InputTensor>& inputs,
    const std::vector<OutputTensor>& outputs,
    std::vector<uint64_t>& compute_input_end_ns,
    std::vector<uint64_t>& compute_output_start_ns, bool distributed)
{
  if (!distributed) {
    LegionTritonRuntime* runtime = model_state_->runtime_;
    runtime->DistributeRunModel(
        model_state_->name, model_state_->version, index_, inputs, outputs,
        compute_input_end_ns, compute_output_start_ns, runtime->rank_, this);
    return;
  }
  AutoBind binding(this);
  model_state_->forward(
      this, index_, runtime_, context_, mapper_, inputs, outputs,
      compute_input_end_ns, compute_output_start_ns);
}

void
LegionModelInstance::ProcessRequests(
    TRITONBACKEND_Request** requests, const uint32_t request_count)
{
  LOG_MESSAGE(
      TRITONSERVER_LOG_VERBOSE,
      (std::string("TRITONBACKEND_ModelExecute: Running ") + Name() + " with " +
       std::to_string(request_count) + " requests")
          .c_str());

  uint64_t request_start_ns = 0;
  SET_TIMESTAMP(request_start_ns);

  const int max_batch_size = Model()->MaxBatchSize();

  // For each request collect the total batch size for this inference
  // execution. The batch-size, number of inputs, and size of each
  // input has already been checked so don't need to do that here.
  size_t total_batch_size = 0;
  std::vector<size_t> request_batch_sizes;
  for (size_t i = 0; i < request_count; i++) {
    // If we get a nullptr request then something is badly wrong. Fail
    // and release all requests.
    if (requests[i] == nullptr) {
      RequestsRespondWithError(
          requests, request_count,
          TRITONSERVER_ErrorNew(
              TRITONSERVER_ERROR_INTERNAL,
              std::string(
                  "null request given to Legion backend for '" + Name() + "'")
                  .c_str()));
      return;
    }

    if (max_batch_size > 0) {
      // Retrieve the batch size from one of the inputs, if the model
      // supports batching, the first dimension size is batch size
      TRITONBACKEND_Input* input;
      TRITONSERVER_Error* err =
          TRITONBACKEND_RequestInputByIndex(requests[i], 0 /* index */, &input);
      if (err == nullptr) {
        const int64_t* shape;
        err = TRITONBACKEND_InputProperties(
            input, nullptr, nullptr, &shape, nullptr, nullptr, nullptr);
        total_batch_size += shape[0];
        request_batch_sizes.emplace_back(shape[0]);
      }
      if (err != nullptr) {
        RequestsRespondWithError(requests, request_count, err);
        return;
      }
    } else {
      total_batch_size += 1;
      request_batch_sizes.emplace_back(1);
    }
  }

  // If there are no valid payloads then no need to run the inference.
  if (total_batch_size == 0) {
    return;
  }

  // Make sure the maximum batch size is not exceeded. The
  // total_batch_size must be 1 for models that don't support batching
  // (i.e. max_batch_size == 0). If max_batch_size is exceeded then
  // scheduler has done something badly wrong so fail and release all
  // requests.
  if ((total_batch_size != 1) && (total_batch_size > (size_t)max_batch_size)) {
    RequestsRespondWithError(
        requests, request_count,
        TRITONSERVER_ErrorNew(
            TRITONSERVER_ERROR_INTERNAL,
            std::string(
                "batch size " + std::to_string(total_batch_size) + " for '" +
                Name() + "', max allowed is " + std::to_string(max_batch_size))
                .c_str()));
    return;
  }

  // At this point we are committed to running inference with all
  // 'requests'. Create a response for each request. During input
  // processing if there is an error with any request that error will
  // be sent immediately with the corresponding response (and the
  // response unique_ptr will then be nullptr). The request object
  // itself will not be released until after all inferencing is done
  // (below) as we may need to access the request object when
  // determine how to process outputs (for example, even if we don't
  // need the outputs for a request that has an error, we do need to
  // know the size of those outputs associated with the request so we
  // can skip them in the output tensors).
  std::vector<TRITONBACKEND_Response*> responses;
  responses.reserve(request_count);

  for (size_t i = 0; i < request_count; i++) {
    TRITONBACKEND_Response* response;
    auto err = TRITONBACKEND_ResponseNew(&response, requests[i]);
    if (err == nullptr) {
      responses.emplace_back(response);
    } else {
      responses.emplace_back(nullptr);
      LOG_MESSAGE(TRITONSERVER_LOG_ERROR, "Fail to create response");
      TRITONSERVER_ErrorDelete(err);
    }
  }

  // Prepare I/O
  std::vector<InputTensor> inputs;
  if (!SetInputTensors(
          total_batch_size, requests, request_count, &responses, inputs)) {
    return;
  }

  std::vector<OutputTensor> outputs;
  if (!SetOutputTensors(
          total_batch_size, request_batch_sizes, requests, request_count,
          &responses, outputs)) {
    return;
  }

  std::vector<uint64_t> compute_input_end_ns(request_count);
  std::vector<uint64_t> compute_output_start_ns(request_count);
  RunModel(inputs, outputs, compute_input_end_ns, compute_output_start_ns);

  uint64_t request_end_ns = request_start_ns;
  SET_TIMESTAMP(request_end_ns);

  // There are two types of statistics that we can report... the
  // statistics for the entire batch of requests that we just executed
  // and statistics for each individual request. Statistics for each
  // individual request were reported above inside the loop as each
  // request was processed (or for failed requests we report that
  // failure below). Here we report statistics for the entire batch of
  // requests.
  LOG_IF_ERROR(
      TRITONBACKEND_ModelInstanceReportBatchStatistics(
          TritonModelInstance(), total_batch_size, request_start_ns,
          compute_input_end_ns.front(), compute_output_start_ns.front(),
          request_end_ns),
      "failed reporting batch request statistics");

  // We could have released each request as soon as we sent the
  // corresponding response. But for clarity we just release them all
  // here. Note that is something goes wrong when releasing a request
  // all we can do is log it... there is no response left to use to
  // report an error.
  for (uint32_t r = 0; r < request_count; ++r) {
    TRITONBACKEND_Request* request = requests[r];

    // If we get to this point then there hasn't been any error and
    // the response is complete and we can send it. This is the last
    // (and only) response that we are sending for the request so we
    // must mark it FINAL. If there is an error when sending all we
    // can do is log it.
    LOG_IF_ERROR(
        TRITONBACKEND_ResponseSend(
            responses[r], TRITONSERVER_RESPONSE_COMPLETE_FINAL,
            nullptr /* success */),
        "failed sending response");

    // Report statistics for the successful request. For an instance
    // using the CPU we don't associate any device with the
    // statistics, otherwise we associate the instance's device.
    LOG_IF_ERROR(
        TRITONBACKEND_ModelInstanceReportStatistics(
            TritonModelInstance(), request, true /* success */,
            request_start_ns, compute_input_end_ns[r],
            compute_output_start_ns[r], request_end_ns),
        "failed reporting request statistics");

    // Before releasing, record failed requests as those where
    // responses[r] is nullptr. The timestamps are ignored in this
    // case.
    if (responses[r] == nullptr) {
      LOG_IF_ERROR(
          TRITONBACKEND_ModelInstanceReportStatistics(
              TritonModelInstance(), request, false /* success */, 0, 0, 0, 0),
          "failed reporting request statistics");
    }

    LOG_IF_ERROR(
        TRITONBACKEND_RequestRelease(request, TRITONSERVER_REQUEST_RELEASE_ALL),
        "failed releasing request");
  }
}

bool
LegionModelInstance::SetInputTensors(
    const size_t total_batch_size, TRITONBACKEND_Request** requests,
    const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    std::vector<InputTensor>& inputs)
{
  // [FIXME] more checking in terms of expected byte size and actual byte size
  const int max_batch_size = Model()->MaxBatchSize();
  size_t padding_batch_size =
      (max_batch_size == 0) ? 0 : max_batch_size - total_batch_size;

  // All requests must have equally-sized input tensors so use any
  // request as the representative for the input tensors.
  uint32_t input_count;
  RESPOND_ALL_AND_RETURN_IF_ERROR(
      false, responses, request_count,
      TRITONBACKEND_RequestInputCount(requests[0], &input_count));
  inputs.resize(input_count);
  LegionTritonRuntime* runtime = model_state_->runtime_;
  for (uint32_t input_idx = 0; input_idx < input_count; input_idx++) {
    InputTensor& tensor = inputs[input_idx];
    TRITONBACKEND_Input* input;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        false, responses, request_count,
        TRITONBACKEND_RequestInputByIndex(requests[0], input_idx, &input));

    const char* input_name;
    TRITONSERVER_DataType input_datatype;
    const int64_t* input_shape;
    uint32_t input_dims_count;
    RESPOND_ALL_AND_RETURN_IF_ERROR(
        false, responses, request_count,
        TRITONBACKEND_InputProperties(
            input, &input_name, &input_datatype, &input_shape,
            &input_dims_count, nullptr, nullptr));

    tensor.name_ = input_name;
    std::vector<int64_t> batchn_shape(
        input_shape, input_shape + input_dims_count);

    tensor.strides_ = std::vector<int64_t>(
        input_dims_count, GetByteSize(input_datatype, {1}));
    if (input_dims_count > 1) {
      for (size_t i = input_dims_count - 1; i > 0; --i) {
        tensor.strides_[i - 1] = tensor.strides_[i] * batchn_shape[i];
      }
    }

    for (size_t request_idx = 0; request_idx < request_count; ++request_idx) {
      TRITONBACKEND_Input* input;
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          false, responses, request_count,
          TRITONBACKEND_RequestInputByIndex(
              requests[request_idx], input_idx, &input));

      uint64_t total_buffer_byte_size;
      uint32_t buffer_count;
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          false, responses, request_count,
          TRITONBACKEND_InputProperties(
              input, nullptr, nullptr, nullptr, nullptr,
              &total_buffer_byte_size, &buffer_count));

      // Check if the input buffers need to be preprocessed into
      // contiguous buffer that satisfies the constraints
      bool need_preprocess = false;
      std::vector<const void*> buffers;
      std::vector<Memory> buffer_memories;
      std::vector<std::pair<TRITONSERVER_MemoryType, int64_t>> buffer_locations;
      // FIXME need to understand how a buffer satisfies the constraints,
      // currently the request input must be in one contiguous buffer, and
      // we shouldn't need to concatenate buffer for requests as splitting along
      // batch dimension should be okay.
      need_preprocess = (buffer_count > 1);
      if (!need_preprocess) {
        const void* buffer;
        uint64_t buffer_byte_size;
        TRITONSERVER_MemoryType memory_type;
        int64_t memory_type_id;
        RESPOND_ALL_AND_RETURN_IF_ERROR(
            false, responses, request_count,
            TRITONBACKEND_InputBuffer(
                input, 0, &buffer, &buffer_byte_size, &memory_type,
                &memory_type_id));
        buffers.emplace_back(buffer);
        buffer_locations.emplace_back(memory_type, memory_type_id);
        buffer_memories.emplace_back(
            runtime->FindMemory(memory_type, memory_type_id));
      }
      // for (size_t buffer_idx = 0; buffer_idx < buffer_count; ++buffer_count)
      // {
      //   const void* buffer;
      //   uint64_t buffer_byte_size;
      //   TRITONSERVER_MemoryType memory_type;
      //   int64_t memory_type_id;
      //   RESPOND_ALL_AND_RETURN_IF_ERROR(
      //       false, responses, request_count,
      //       TRITONBACKEND_InputBuffer(
      //           input, buffer_idx, &buffer, &buffer_byte_size, &memory_type,
      //           &memory_type_id));
      //   // Check if the buffer is good
      //   for (auto it = tensor.strides_.cbegin();
      //        it != tensor.strides_.cend(); ++it) {
      //     if (*it <= buffer_byte_size) {
      //       need_preprocess = ((buffer_byte_size % *it) != 0);
      //       break;
      //     }
      //   }
      //   if (need_preprocess) {
      //     break;
      //   } else {
      //     buffers.emplace_back(buffer);
      //     buffer_locations.emplace_back(memory_type, memory_type_id);
      //   }
      // }
      if (need_preprocess) {
        // FIXME using CPU for now, can be smart based on what kind of input
        // buffer that the model prefers
        BackendMemory* backend_memory;
        RESPOND_ALL_AND_RETURN_IF_ERROR(
            false, responses, request_count,
            BackendMemory::Create(
                Model()->TritonMemoryManager(),
                BackendMemory::AllocationType::CPU, 0, total_buffer_byte_size,
                &backend_memory));
        tensor.allocated_memory_.emplace_back(backend_memory);
        RESPOND_ALL_AND_RETURN_IF_ERROR(
            false, responses, request_count,
            ReadInputTensor(
                requests[request_idx], input_name, backend_memory->MemoryPtr(),
                &total_buffer_byte_size));
        tensor.buffers_.emplace_back(backend_memory->MemoryPtr());
        tensor.buffer_locations_.emplace_back(
            backend_memory->MemoryType(), backend_memory->MemoryTypeId());
        tensor.buffer_memories_.emplace_back(runtime->FindMemory(
            backend_memory->MemoryType(), backend_memory->MemoryTypeId()));
      } else {
        std::copy(
            buffers.begin(), buffers.end(),
            std::back_inserter(tensor.buffers_));
        std::copy(
            buffer_locations.begin(), buffer_locations.end(),
            std::back_inserter(tensor.buffer_locations_));
        std::copy(
            buffer_memories.begin(), buffer_memories.end(),
            std::back_inserter(tensor.buffer_memories_));
      }
    }

    if (padding_batch_size != 0) {
      size_t byte_size = tensor.strides_[0] * padding_batch_size;
      BackendMemory* backend_memory;
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          false, responses, request_count,
          BackendMemory::Create(
              Model()->TritonMemoryManager(),
              BackendMemory::AllocationType::CPU, 0, byte_size,
              &backend_memory));
      tensor.allocated_memory_.emplace_back(backend_memory);
      // set the value of the padding to zeros
      memset(backend_memory->MemoryPtr(), 0, byte_size);
      tensor.buffers_.emplace_back(backend_memory->MemoryPtr());
      tensor.buffer_locations_.emplace_back(
          backend_memory->MemoryType(), backend_memory->MemoryTypeId());
    }
  }
  return true;
}

bool
LegionModelInstance::SetOutputTensors(
    const size_t total_batch_size,
    const std::vector<size_t>& request_batch_sizes,
    TRITONBACKEND_Request** requests, const uint32_t request_count,
    std::vector<TRITONBACKEND_Response*>* responses,
    std::vector<OutputTensor>& outputs)
{
  const int max_batch_size = Model()->MaxBatchSize();
  size_t padding_batch_size =
      (max_batch_size == 0) ? 0 : max_batch_size - total_batch_size;

  const auto& output_infos = model_state_->OutputInfos();
  outputs.reserve(output_infos.size());
  LegionTritonRuntime* runtime = model_state_->runtime_;
  for (const auto& output_info : output_infos) {
    outputs.emplace_back();
    OutputTensor& tensor = outputs.back();
    tensor.name_ = std::get<0>(output_info);
    const auto& triton_dtype = std::get<1>(output_info);
    // Make a copy of it as the batch dimension will be updated to
    // match individual request batch size.
    std::vector<int64_t> batchn_shape = std::get<2>(output_info);

    tensor.strides_ = std::vector<int64_t>(
        batchn_shape.size(), GetByteSize(triton_dtype, {1}));
    if (batchn_shape.size() > 1) {
      for (size_t i = (batchn_shape.size() - 1); i > 0; --i) {
        tensor.strides_[i - 1] = tensor.strides_[i] * batchn_shape[i];
      }
    }
    size_t batch1_byte_size = GetByteSize(triton_dtype, batchn_shape);
    if (max_batch_size != 0) {
      batch1_byte_size /= batchn_shape[0];
    }
    // Prepare the output buffer for each response, if the output is not
    // requested, backend managed buffer will be used
    for (size_t request_idx = 0; request_idx < request_count; ++request_idx) {
      uint32_t requested_output_count;
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          false, responses, request_count,
          TRITONBACKEND_RequestOutputCount(
              requests[request_idx], &requested_output_count));
      bool found = false;
      for (size_t output_idx = 0; output_idx < requested_output_count;
           ++output_idx) {
        const char* output_name;
        RESPOND_ALL_AND_RETURN_IF_ERROR(
            false, responses, request_count,
            TRITONBACKEND_RequestOutputName(
                requests[request_idx], output_idx, &output_name));
        if (tensor.name_ == output_name) {
          found = true;
          break;
        }
      }
      if (found) {
        if (max_batch_size != 0) {
          batchn_shape[0] = request_batch_sizes[request_idx];
        }
        TRITONBACKEND_Output* response_output;
        RESPOND_ALL_AND_RETURN_IF_ERROR(
            false, responses, request_count,
            TRITONBACKEND_ResponseOutput(
                (*responses)[request_idx], &response_output,
                tensor.name_.c_str(), triton_dtype, batchn_shape.data(),
                batchn_shape.size()));
        void* buffer;
        // FIXME using CPU for now, can be smart based on what kind of output
        // buffer that the model produce
        TRITONSERVER_MemoryType memory_type = TRITONSERVER_MEMORY_CPU;
        int64_t memory_type_id = 0;
        RESPOND_ALL_AND_RETURN_IF_ERROR(
            false, responses, request_count,
            TRITONBACKEND_OutputBuffer(
                response_output, &buffer,
                batch1_byte_size * request_batch_sizes[request_idx],
                &memory_type, &memory_type_id));
        tensor.buffers_.emplace_back(buffer);
        tensor.buffer_locations_.emplace_back(memory_type, memory_type_id);
        tensor.buffer_memories_.emplace_back(
            runtime->FindMemory(memory_type, memory_type_id));
      } else {
        BackendMemory* backend_memory;
        RESPOND_ALL_AND_RETURN_IF_ERROR(
            false, responses, request_count,
            BackendMemory::Create(
                Model()->TritonMemoryManager(),
                BackendMemory::AllocationType::CPU, 0,
                batch1_byte_size * request_batch_sizes[request_idx],
                &backend_memory));
        tensor.allocated_memory_.emplace_back(backend_memory);
        tensor.buffers_.emplace_back(backend_memory->MemoryPtr());
        tensor.buffer_locations_.emplace_back(
            backend_memory->MemoryType(), backend_memory->MemoryTypeId());
        tensor.buffer_memories_.emplace_back(runtime->FindMemory(
            backend_memory->MemoryType(), backend_memory->MemoryTypeId()));
      }
    }
    if (padding_batch_size != 0) {
      BackendMemory* backend_memory;
      RESPOND_ALL_AND_RETURN_IF_ERROR(
          false, responses, request_count,
          BackendMemory::Create(
              Model()->TritonMemoryManager(),
              BackendMemory::AllocationType::CPU, 0,
              batch1_byte_size * padding_batch_size, &backend_memory));
      tensor.allocated_memory_.emplace_back(backend_memory);
      tensor.buffers_.emplace_back(backend_memory->MemoryPtr());
      tensor.buffer_locations_.emplace_back(
          backend_memory->MemoryType(), backend_memory->MemoryTypeId());
      tensor.buffer_memories_.emplace_back(runtime->FindMemory(
          backend_memory->MemoryType(), backend_memory->MemoryTypeId()));
    }
  }
  return true;
}

IndexSpace
LegionModelInstance::find_or_create_index_space(const Domain& domain)
{
  std::map<Domain, IndexSpace>::const_iterator finder =
      top_level_index_spaces.find(domain);
  if (finder != top_level_index_spaces.end())
    return finder->second;
  IndexSpace result = runtime_->create_index_space(context_, domain);
  top_level_index_spaces[domain] = result;
  return result;
}

IndexPartition
LegionModelInstance::find_or_create_partition(
    IndexSpace top_level_space, IndexSpace color_space,
    const DomainTransform& part_transform, const Domain& part_extent,
    PartitionKind kind)
{
  std::map<IndexSpace, std::vector<Partition>>::const_iterator finder =
      top_level_partitions.find(top_level_space);
  if (finder != top_level_partitions.end()) {
    switch (part_extent.get_dim()) {
#define DIMFUNC(DIM)                                                         \
  case DIM: {                                                                \
    Transform<DIM, DIM> transform = part_transform;                          \
    Rect<DIM> extent = part_extent;                                          \
    for (std::vector<Partition>::const_iterator it = finder->second.begin(); \
         it != finder->second.end(); it++) {                                 \
      if (color_space != it->color_space)                                    \
        continue;                                                            \
      Rect<DIM> prev_extent = it->extent;                                    \
      if (extent != prev_extent)                                             \
        continue;                                                            \
      Transform<DIM, DIM> prev_transform = it->transform;                    \
      bool isomorphic = true;                                                \
      for (int d = 0; d < DIM; d++) {                                        \
        if (transform[d] == prev_transform[d])                               \
          continue;                                                          \
        isomorphic = false;                                                  \
        break;                                                               \
      }                                                                      \
      if (!isomorphic)                                                       \
        continue;                                                            \
      return it->partition;                                                  \
    }                                                                        \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
      default:
        assert(false);
    }
  }
  // If we get here then we need to make it
  IndexPartition result = runtime_->create_partition_by_restriction(
      context_, top_level_space, color_space, part_transform, part_extent,
      kind);
  // Save it for later
  top_level_partitions[top_level_space].push_back(
      Partition(color_space, result, part_transform, part_extent));
  return result;
}

FieldSpace
LegionModelInstance::find_or_create_field_space(DataType data_type)
{
  std::map<DataType, FieldSpace>::const_iterator finder =
      top_level_field_spaces.find(data_type);
  if (finder != top_level_field_spaces.end())
    return finder->second;
  // make a new field space
  FieldSpace result = runtime_->create_field_space(context_);
  top_level_field_spaces[data_type] = result;
  // Allocate a field of the right size in the field space
  FieldAllocator allocator = runtime_->create_field_allocator(context_, result);
  allocator.allocate_field(sizeof_datatype(data_type), FID_DATA);
  return result;
}

LogicalRegion
LegionModelInstance::create_tensor_region(Tensor* tensor)
{
  assert(!tensor->region[index_].exists());
  DomainPoint lo, hi;
  lo.dim = tensor->bounds.size();
  hi.dim = tensor->bounds.size();
  for (unsigned d = 0; d < tensor->bounds.size(); d++) {
    lo[d] = 0;
    assert(tensor->bounds[d] > 0);
    hi[d] = tensor->bounds[d] - 1;  // legion domains are inclusive
  }
  Domain bounds(lo, hi);
  IndexSpace is = find_or_create_index_space(bounds);
  FieldSpace fs = find_or_create_field_space(tensor->type);
  LogicalRegion lr = runtime_->create_logical_region(context_, is, fs);
  // Save the handle in the tensor
  tensor->region[index_] = lr;
  // Remember the name for when we need to delete it
  top_level_regions.push_back(lr);
  return lr;
}

LogicalPartition
LegionModelInstance::find_or_create_tiled_partition(
    Tensor* tensor, const LayerStrategy* strategy)
{
  assert(tensor->region[index_].exists());
  if (tensor->partition[index_].exists())
    return tensor->partition[index_];
  assert(size_t(strategy->nDims) == tensor->bounds.size());
  Domain color_domain = strategy->get_launch_domain();
  IndexSpace color_space = find_or_create_index_space(color_domain);
  Domain part_extent;
  DomainTransform part_transform;
  switch (tensor->bounds.size()) {
#define DIMFUNC(DIM)                                           \
  case DIM: {                                                  \
    Point<DIM> ext_hi;                                         \
    Rect<DIM> color_rect = color_domain;                       \
    for (int d = 0; d < DIM; d++) {                            \
      size_t parts = color_rect.hi[d] - color_rect.lo[d] + 1;  \
      ext_hi[d] = (tensor->bounds[d] + parts - 1) / parts - 1; \
    }                                                          \
    Rect<DIM> extent(Point<DIM>::ZEROES(), ext_hi);            \
    Transform<DIM, DIM> transform;                             \
    for (int i = 0; i < DIM; i++)                              \
      for (int j = 0; j < DIM; j++)                            \
        if (i == j)                                            \
          transform[i][j] = extent.hi[i] - extent.lo[i] + 1;   \
        else                                                   \
          transform[i][j] = 0;                                 \
    part_extent = extent;                                      \
    part_transform = transform;                                \
    break;                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  // Find or compute the index partition
  LogicalRegion region = tensor->region[index_];
  IndexPartition partition = find_or_create_partition(
      region.get_index_space(), color_space, part_transform, part_extent,
      LEGION_DISJOINT_COMPLETE_KIND);
  LogicalPartition result = runtime_->get_logical_partition_by_tree(
      context_, partition, region.get_field_space(), region.get_tree_id());
  tensor->partition[index_] = result;
  return result;
}

}}}  // namespace triton::backend::legion
