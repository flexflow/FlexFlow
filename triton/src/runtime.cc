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

#include "runtime.h"
#include "instance.h"
#include "legion/legion_utilities.h"
#include "model.h"
#include "operator.h"

using namespace Legion;

namespace triton { namespace backend { namespace legion {

TRITONSERVER_Error*
LegionTritonRuntime::Create(Legion::TaskID ttid, LegionTritonRuntime** runtime)
{
  Machine machine = Machine::get_machine();
  // Find our local utility processor first
  Processor local_proc;
  std::vector<Processor> local_cpus, local_gpus;
  {
    Machine::ProcessorQuery query(machine);
    query.only_kind(Processor::LOC_PROC /*CPU*/);
    query.local_address_space();
    size_t count = query.count();
    if (count == 0)
      return TRITONSERVER_ErrorNew(
          TRITONSERVER_ERROR_UNKNOWN,
          "Unable to find any Realm CPU processors");
    if (count > 1) {
      local_cpus.reserve(count);
      for (Machine::ProcessorQuery::iterator it = query.begin();
           it != query.end(); it++)
        local_cpus.push_back(*it);
      local_proc = ProcessorGroup::create_group(local_cpus);
    } else {
      local_proc = query.first();
      local_cpus.push_back(local_proc);
    }
  }
  const AddressSpaceID local_space = local_proc.address_space();
  // Find a remote utility processor to use as well for each node
  std::vector<Processor> remote_procs, all_cpus, all_gpus;
  {
    Machine::ProcessorQuery query(machine);
    query.only_kind(Processor::LOC_PROC /*CPU*/);
    std::map<AddressSpaceID, Processor> unique_spaces;
    all_cpus.reserve(query.count());
    for (Machine::ProcessorQuery::iterator it = query.begin();
         it != query.end(); it++) {
      all_cpus.push_back(*it);
      AddressSpaceID space = it->address_space();
      if (space == local_space)
        continue;
      if (unique_spaces.find(space) != unique_spaces.end())
        continue;
      unique_spaces[space] = *it;
    }
    // plus one because we did not include oursevles
    remote_procs.resize(unique_spaces.size() + 1, Processor::NO_PROC);
    for (auto it = unique_spaces.begin(); it != unique_spaces.end(); it++)
      remote_procs[it->first] = it->second;
  }
  {
    Machine::ProcessorQuery query(machine);
    query.only_kind(Processor::TOC_PROC /*GPU*/);
    all_gpus.reserve(query.count());
    for (Machine::ProcessorQuery::iterator it = query.begin();
         it != query.end(); it++) {
      all_gpus.push_back(*it);
      if (it->address_space() == local_space)
        local_gpus.push_back(*it);
    }
  }
  *runtime = new LegionTritonRuntime(
      Runtime::get_runtime(), ttid, local_space, remote_procs.size(),
      local_proc, std::move(remote_procs), std::move(all_cpus),
      std::move(all_gpus), local_gpus);
  // Register our tasks with Realm for handling messages
  std::vector<Realm::Event> ready_events;
  Realm::ProfilingRequestSet no_requests;
  CodeDescriptor message_desc(LegionTritonRuntime::InstanceMessageTask);
  CodeDescriptor context_desc(LegionTritonRuntime::CreateContextTask);
  CodeDescriptor inference_desc(LegionTritonRuntime::RunModelInferenceTask);
  CodeDescriptor load_layer_desc(LegionTritonRuntime::LoadLayerTask);
  CodeDescriptor free_layer_desc(LegionTritonRuntime::FreeLayerTask);
  for (auto proc : local_cpus) {
    Realm::Event registered = proc.register_task(
        INSTANCE_CREATE_TASK_ID, message_desc, no_requests, runtime,
        sizeof(LegionTritonRuntime*));
    if (registered.exists())
      ready_events.push_back(registered);
    registered = proc.register_task(
        CONTEXT_CREATE_TASK_ID, context_desc, no_requests, runtime,
        sizeof(LegionTritonRuntime*));
    if (registered.exists())
      ready_events.push_back(registered);
    registered = proc.register_task(
        RUN_MODEL_INFERENCE_TASK_ID, inference_desc, no_requests, runtime,
        sizeof(LegionTritonRuntime*));
    if (registered.exists())
      ready_events.push_back(registered);
    registered = proc.register_task(
        LOAD_LAYER_TASK_ID, load_layer_desc, no_requests, runtime,
        sizeof(LegionTritonRuntime*));
    if (registered.exists())
      ready_events.push_back(registered);
    registered = proc.register_task(
        FREE_LAYER_TASK_ID, free_layer_desc, no_requests, runtime,
        sizeof(LegionTritonRuntime*));
    if (registered.exists())
      ready_events.push_back(registered);
  }
  // also need to register layer tasks on GPUs as well
  CodeDescriptor init_cuda_desc(LegionTritonRuntime::InitCudaTask);
  unsigned gpu_index = 0;
  for (auto proc : local_gpus) {
    Realm::Event registered = proc.register_task(
        INIT_CUDALIBS_TASK_ID, init_cuda_desc, no_requests, runtime,
        sizeof(LegionTritonRuntime*));
    // Also launch the task to initialize the cuda libraries
    registered = proc.spawn(
        INIT_CUDALIBS_TASK_ID, &gpu_index, sizeof(gpu_index), registered);
    gpu_index++;
    if (!registered.exists())
      ready_events.push_back(registered);
    registered = proc.register_task(
        LOAD_LAYER_TASK_ID, load_layer_desc, no_requests, runtime,
        sizeof(LegionTritonRuntime*));
    if (registered.exists())
      ready_events.push_back(registered);
    registered = proc.register_task(
        FREE_LAYER_TASK_ID, free_layer_desc, no_requests, runtime,
        sizeof(LegionTritonRuntime*));
    if (registered.exists())
      ready_events.push_back(registered);
  }
  // Do a Realm barrier here to make sure everyone is done registering
  Realm::Runtime realm = Realm::Runtime::get_runtime();
  if (!ready_events.empty())
    realm
        .collective_spawn_by_kind(
            Realm::Processor::LOC_PROC /*CPU*/,
            Realm::Processor::TASK_ID_PROCESSOR_NOP, nullptr, 0,
            true /*one per node*/, Realm::Event::merge_events(ready_events))
        .external_wait();
  else
    realm
        .collective_spawn_by_kind(
            Realm::Processor::LOC_PROC /*CPU*/,
            Realm::Processor::TASK_ID_PROCESSOR_NOP, nullptr, 0,
            true /*one per node*/)
        .external_wait();

  return nullptr;
}

LegionTritonRuntime::LegionTritonRuntime(
    Legion::Runtime* lg, Legion::TaskID ttid, Legion::AddressSpaceID rank,
    size_t total, Processor local, std::vector<Processor>&& remote,
    std::vector<Processor>&& cpus, std::vector<Processor>&& gpus,
    const std::vector<Processor>& local_gpus, bool allowTensorOpMathConv)
    : legion_(lg), top_task_id_(ttid), rank_(rank), total_ranks_(total),
      local_proc_(local), local_sysmem_(FindLocalSysmem()),
      local_regmem_(FindLocalRegmem()), remote_procs_(std::move(remote)),
      all_cpus_(std::move(cpus)), all_gpus_(std::move(gpus)),
      local_cpus_(FindLocal(rank_, all_cpus_)),
      local_gpus_(FindLocal(rank_, all_gpus_)),
      local_framebuffers_(FindLocalFramebuffers(local_gpus)),
      allowTensorOpMathConversion_(allowTensorOpMathConv)
{
  creation_barrier_ = Realm::Barrier::NO_BARRIER;
  for (unsigned idx = 0; idx < local_gpus.size(); idx++)
    gpu_lookup_[local_gpus[idx]] = idx;
}

/*static*/ std::vector<Realm::Processor>
LegionTritonRuntime::FindLocal(
    AddressSpaceID local, const std::vector<Realm::Processor>& procs)
{
  std::vector<Realm::Processor> result;
  for (auto proc : procs)
    if (proc.address_space() == local)
      result.push_back(proc);
  return result;
}

/*static*/ Realm::Memory
LegionTritonRuntime::FindLocalSysmem(void)
{
  Machine machine = Machine::get_machine();
  Machine::MemoryQuery local_sysmem(machine);
  local_sysmem.local_address_space();
  local_sysmem.only_kind(Memory::SYSTEM_MEM);
  assert(local_sysmem.count() == 1);
  return local_sysmem.first();
}

/*static*/ Realm::Memory
LegionTritonRuntime::FindLocalRegmem(void)
{
  Machine machine = Machine::get_machine();
  Machine::MemoryQuery local_regmem(machine);
  local_regmem.local_address_space();
  local_regmem.only_kind(Memory::REGDMA_MEM);
  assert(local_regmem.count() <= 1);
  if (local_regmem.count() == 0)
    return Memory::NO_MEMORY;
  return local_regmem.first();
}

/*static*/ std::vector<Realm::Memory>
LegionTritonRuntime::FindLocalFramebuffers(
    const std::vector<Realm::Processor>& gpus)
{
  Machine machine = Machine::get_machine();
  std::vector<Realm::Memory> framebuffers(gpus.size());
  for (unsigned idx = 0; idx < gpus.size(); idx++) {
    // Now we can find our framebuffer
    Machine::MemoryQuery local_fb(machine);
    local_fb.only_kind(Memory::GPU_FB_MEM);
    local_fb.best_affinity_to(gpus[idx]);
    assert(local_fb.count() >= 1);
    framebuffers[idx] = local_fb.first();
  }
  return framebuffers;
}

void
LegionTritonRuntime::RecordModel(LegionModelState* model)
{
  AutoLock<true> lock(lock_);
  models_.push_back(model);
}

void
LegionTritonRuntime::RemoveModel(LegionModelState* model)
{
  AutoLock<true> lock(lock_);
  for (auto it = models_.begin(); it != models_.end(); it++) {
    if ((*it) != model)
      continue;
    models_.erase(it);
    break;
  }
}

void
LegionTritonRuntime::RendezvousContextCreation(
    LegionModelInstance* instance, Realm::UserEvent ready)
{
  // If we're not rank 0 send the message there
  if (rank_ > 0) {
    Serializer rez;
    const size_t length = instance->model_state_->name.size();
    rez.serialize(length);
    rez.serialize(instance->model_state_->name.c_str(), length);
    rez.serialize(instance->model_state_->version);
    rez.serialize(instance->index_);
    rez.serialize(local_proc_);
    rez.serialize(instance);
    rez.serialize(ready);
    remote_procs_[0].spawn(
        INSTANCE_CREATE_TASK_ID, rez.get_buffer(), rez.get_used_bytes(),
        Realm::Event::NO_EVENT, INT_MAX /*high priority*/);
  } else
    HandleContextCreation(
        instance->model_state_->name, instance->model_state_->version,
        instance->index_, local_proc_, instance, ready);
}

unsigned
LegionTritonRuntime::FindGPUIndex(Processor proc) const
{
  std::map<Processor, unsigned>::const_iterator finder = gpu_lookup_.find(proc);
  assert(finder != gpu_lookup_.end());
  return finder->second;
}

const std::vector<Processor>&
LegionTritonRuntime::FindAllProcessors(Processor::Kind kind)
{
  switch (kind) {
    case Processor::LOC_PROC:
      return all_cpus_;
    case Processor::TOC_PROC:
      return all_gpus_;
    default:
      abort();
  }
  return all_cpus_;
}

const std::vector<Processor>&
LegionTritonRuntime::FindLocalProcessors(Processor::Kind kind)
{
  switch (kind) {
    case Processor::LOC_PROC:
      return local_cpus_;
    case Processor::TOC_PROC:
      return local_gpus_;
    default:
      abort();
  }
  return local_cpus_;
}

Memory
LegionTritonRuntime::FindMemory(
    TRITONSERVER_MemoryType type, uint64_t device_id)
{
  switch (type) {
    case TRITONSERVER_MEMORY_CPU: {
      assert(local_sysmem_.exists());
      return local_sysmem_;
    }
    case TRITONSERVER_MEMORY_CPU_PINNED: {
      assert(local_regmem_.exists());
      return local_regmem_;
    }
    case TRITONSERVER_MEMORY_GPU: {
      // Hopefully Triton counts device IDs the same way as Realm does
      assert(device_id < local_framebuffers_.size());
      return local_framebuffers_[device_id];
    }
    default:
      assert(false);
  }
  return Memory::NO_MEMORY;
}

Realm::Event
LegionTritonRuntime::LoadLayer(Processor proc, Operator* op)
{
  return proc.spawn(
      LOAD_LAYER_TASK_ID, &op, sizeof(op), Realm::Event::NO_EVENT,
      INT_MAX /*high priority*/);
}

Realm::Event
LegionTritonRuntime::FreeLayer(Processor proc, Operator* op)
{
  return proc.spawn(
      FREE_LAYER_TASK_ID, &op, sizeof(op), Realm::Event::NO_EVENT,
      INT_MAX /*high priority*/);
}

void
LegionTritonRuntime::HandleContextCreation(
    const std::string& name, uint64_t version, unsigned index,
    Realm::Processor source, LegionModelInstance* instance,
    Realm::UserEvent ready, bool external, bool need_lock)
{
  // Should only be here on rank 0
  assert(rank_ == 0);
  if (need_lock) {
    if (external) {
      AutoLock<true> lock(lock_);
      HandleContextCreation(
          name, version, index, source, instance, ready, true, false);
    } else {
      AutoLock<false> lock(lock_);
      HandleContextCreation(
          name, version, index, source, instance, ready, false, false);
    }
    return;
  }
  // Now we've got the lock
  for (auto it = pending_contexts_.begin(); it != pending_contexts_.end();
       it++) {
    if (!it->matches(name, version, index))
      continue;
    assert(it->requests.find(source) == it->requests.end());
    it->requests[source] = std::make_pair(instance, ready);
    // If we've seen all the requests we know we're ready for the rendezvous
    if (it->requests.size() == total_ranks_) {
      CreatePendingContext(*it);
      pending_contexts_.erase(it);
    }
    return;
  }
  if (total_ranks_ > 1) {
    pending_contexts_.emplace_back(PendingContext(name, version, index));
    pending_contexts_.back().requests[source] = std::make_pair(instance, ready);
  } else {
    // Special case of single process execution
    PendingContext pending(name, version, index);
    pending.requests[source] = std::make_pair(instance, ready);
    CreatePendingContext(pending);
  }
}

void
LegionTritonRuntime::CreatePendingContext(const PendingContext& pending)
{
  assert(pending.requests.size() == total_ranks_);
  // Get a barrier for coordinating ordering of the creations
  Realm::Barrier bar;
  // Also get the precondition from the previous creation
  Realm::Event precondition;
  if (creation_barrier_.exists()) {
    bar = creation_barrier_;
    precondition = creation_barrier_.get_previous_phase();
  } else {
    creation_barrier_ = Realm::Barrier::create_barrier(total_ranks_);
    bar = creation_barrier_;
    precondition = Realm::Event::NO_EVENT;
  }
  creation_barrier_ = creation_barrier_.advance_barrier();
  // TODO: we assert if we run out of barrier generations here
  // If we make more than 2^12 contexts at any point this could be an issue
  // Subject to changes in Realm's setting for default barrier generations
  assert(creation_barrier_.exists());
  // Launch tasks to make the contexts on all the nodes
  // TODO: build a broadcast tree to do this for more scalability
  // For node counts less than a few hundred we should be alright
  for (auto it = pending.requests.begin(); it != pending.requests.end(); it++) {
    Serializer rez;
    rez.serialize(it->second.first);
    rez.serialize(bar);
    it->first.spawn(
        CONTEXT_CREATE_TASK_ID, rez.get_buffer(), rez.get_used_bytes(),
        precondition, INT_MAX /*high priority*/);
    // We know the context will be ready when the barrier completes
    it->second.second.trigger(bar);
  }
}

/*static*/ void
LegionTritonRuntime::InstanceMessageTask(
    const void* args, size_t arglen, const void* data, size_t datalen,
    Realm::Processor p)
{
  assert(datalen == sizeof(LegionTritonRuntime*));
  LegionTritonRuntime* runtime = *((LegionTritonRuntime**)data);
  Deserializer derez(args, arglen);
  size_t length;
  derez.deserialize(length);
  std::string name((const char*)derez.get_current_pointer(), length);
  derez.advance_pointer(length);
  uint64_t version;
  derez.deserialize(version);
  unsigned index;
  derez.deserialize(index);
  Processor source;
  derez.deserialize(source);
  LegionModelInstance* instance;
  derez.deserialize(instance);
  Realm::UserEvent ready;
  derez.deserialize(ready);
  assert(!derez.get_remaining_bytes());

  runtime->HandleContextCreation(
      name, version, index, source, instance, ready, false /*external*/);
}

/*static*/ void
LegionTritonRuntime::CreateContextTask(
    const void* args, size_t arglen, const void* data, size_t datalen,
    Realm::Processor p)
{
  assert(datalen == sizeof(LegionTritonRuntime*));
  LegionTritonRuntime* runtime = *((LegionTritonRuntime**)data);
  Deserializer derez(args, arglen);
  LegionModelInstance* instance;
  derez.deserialize(instance);
  Realm::Barrier barrier;
  derez.deserialize(barrier);
  assert(!derez.get_remaining_bytes());

  // Create the context
  instance->CreateContext(
      runtime->legion_, runtime->top_task_id_, runtime->rank_,
      runtime->total_ranks_, barrier,
      (runtime->InstanceOwner(instance->index_) == runtime->rank_));
  // Arrive on the barrier signaling that we are done
  barrier.arrive();
}

AddressSpaceID
LegionTritonRuntime::InstanceOwner(unsigned instance_index) const
{
  // Round robin instances across the nodes so we get some load balance
  // of responsibilities for scheduling work on different instances
  // TODO: a better hasing scheme here for load balance that incorporates
  // the model names and versions as well
  return (instance_index % total_ranks_);
}

LegionModelInstance*
LegionTritonRuntime::FindModelInstance(
    const std::string& model_name, uint64_t model_version,
    unsigned instance_index, bool external, bool need_lock)
{
  if (need_lock) {
    if (external) {
      AutoLock<true> lock(lock_, false /*exclusive*/);
      return FindModelInstance(
          model_name, model_version, instance_index, true, false);
    } else {
      AutoLock<false> lock(lock_, false /*exclusive*/);
      return FindModelInstance(
          model_name, model_version, instance_index, false, false);
    }
  }
  for (auto model : models_) {
    if (model_name != model->name)
      continue;
    if (model_version != model->version)
      continue;
    return model->FindInstance(instance_index, external);
  }
  // should never get here
  assert(false);
  return NULL;
}

void
LegionTritonRuntime::DistributeRunModel(
    const std::string& model_name, uint64_t model_version,
    unsigned instance_index, const std::vector<InputTensor>& inputs,
    const std::vector<OutputTensor>& outputs,
    std::vector<uint64_t>& compute_input_end_ns,
    std::vector<uint64_t>& compute_output_start_ns, AddressSpaceID source,
    LegionModelInstance* instance, Realm::Barrier barrier,
    Realm::UserEvent pretrigger, Realm::UserEvent posttrigger)
{
  Realm::Event precondition = Realm::Event::NO_EVENT;
  const AddressSpaceID owner_rank = InstanceOwner(instance_index);
  if (owner_rank != rank_) {
    assert(!pretrigger.exists());
    assert(!posttrigger.exists());
    // Check to see if this is already our broadcast message
    if (source == rank_) {
      assert(!barrier.exists());
      // Send the message to the owner rank to handle this
      pretrigger = Realm::UserEvent::create_user_event();
      posttrigger = Realm::UserEvent::create_user_event();
      Serializer rez;
      PackRunModel(
          rez, model_name, model_version, instance_index, inputs, outputs);
      rez.serialize(Realm::Barrier::NO_BARRIER);
      rez.serialize(pretrigger);
      rez.serialize(posttrigger);
      rez.serialize(rank_);
      remote_procs_[owner_rank].spawn(
          RUN_MODEL_INFERENCE_TASK_ID, rez.get_buffer(), rez.get_used_bytes(),
          Realm::Event::NO_EVENT, INT_MAX);
      precondition = pretrigger;
    } else {
      // This came from the broadcast node so we can just run it
      assert(barrier.exists());
      assert(!pretrigger.exists());
      assert(!posttrigger.exists());
      // No precondition as it was added to the launch
    }
  } else {
    assert(!barrier.exists());
    if (instance == NULL)
      instance = FindModelInstance(
          model_name, model_version, instance_index, (source == rank_));
    barrier = instance->GetExecutionBarrier(
        total_ranks_, precondition, (source == rank_));
    Serializer rez;
    PackRunModel(
        rez, model_name, model_version, instance_index, inputs, outputs);
    rez.serialize(barrier);
    rez.serialize(Realm::UserEvent::NO_USER_EVENT);
    rez.serialize(Realm::UserEvent::NO_USER_EVENT);
    rez.serialize(rank_);
    // Broadcast out the request to all the other ranks
    for (AddressSpaceID rank = 0; rank < total_ranks_; rank++) {
      if (rank == rank_)
        continue;
      if (rank == source) {
        // No need to send a message back to the source
        assert(pretrigger.exists());
        assert(posttrigger.exists());
        pretrigger.trigger(precondition);
        barrier.arrive(1 /*count*/, posttrigger);
      } else
        remote_procs_[rank].spawn(
            RUN_MODEL_INFERENCE_TASK_ID, rez.get_buffer(), rez.get_used_bytes(),
            precondition, INT_MAX);
    }
  }
  // Find the instance we are running
  if (instance == NULL)
    instance = FindModelInstance(
        model_name, model_version, instance_index, (source == rank_));
  // If we have a precondition, wait for that before we start the call
  if (precondition.exists()) {
    if (source == rank_)
      precondition.external_wait();
    else
      precondition.wait();
  }
  // Run the model
  instance->RunModel(
      inputs, outputs, compute_input_end_ns, compute_output_start_ns,
      true /*distributed*/);
  if (!barrier.exists()) {
    assert(posttrigger.exists());
    posttrigger.trigger();
  } else
    barrier.arrive();
}

/*static*/ void
LegionTritonRuntime::PackRunModel(
    Serializer& rez, const std::string& model_name, uint64_t model_version,
    unsigned instance_index, const std::vector<InputTensor>& inputs,
    const std::vector<OutputTensor>& outputs)
{
  const size_t length = model_name.size();
  rez.serialize(length);
  rez.serialize(model_name.c_str(), length);
  rez.serialize(model_version);
  rez.serialize(instance_index);
  rez.serialize<size_t>(inputs.size());
  for (auto& tensor : inputs) {
    const size_t namelen = tensor.name_.size();
    rez.serialize(namelen);
    rez.serialize(tensor.name_.c_str(), namelen);
    rez.serialize<size_t>(tensor.buffers_.size());
    for (auto ptr : tensor.buffers_) rez.serialize(ptr);
    assert(tensor.buffers_.size() == tensor.buffer_locations_.size());
    for (auto& loc : tensor.buffer_locations_) {
      rez.serialize(loc.first);
      rez.serialize(loc.second);
    }
    assert(tensor.buffers_.size() == tensor.buffer_memories_.size());
    for (auto& mem : tensor.buffer_memories_) rez.serialize(mem);
    rez.serialize<size_t>(tensor.strides_.size());
    for (auto stride : tensor.strides_) rez.serialize(stride);
  }
  rez.serialize<size_t>(outputs.size());
  for (auto& tensor : outputs) {
    const size_t namelen = tensor.name_.size();
    rez.serialize(namelen);
    rez.serialize(tensor.name_.c_str(), namelen);
    rez.serialize<size_t>(tensor.buffers_.size());
    for (auto ptr : tensor.buffers_) rez.serialize(ptr);
    assert(tensor.buffers_.size() == tensor.buffer_locations_.size());
    for (auto& loc : tensor.buffer_locations_) {
      rez.serialize(loc.first);
      rez.serialize(loc.second);
    }
    assert(tensor.buffers_.size() == tensor.buffer_memories_.size());
    for (auto& mem : tensor.buffer_memories_) rez.serialize(mem);
    rez.serialize<size_t>(tensor.strides_.size());
    for (auto stride : tensor.strides_) rez.serialize(stride);
  }
}

/*static*/ void
LegionTritonRuntime::RunModelInferenceTask(
    const void* args, size_t arglen, const void* data, size_t datalen,
    Realm::Processor p)
{
  assert(datalen == sizeof(LegionTritonRuntime*));
  LegionTritonRuntime* runtime = *((LegionTritonRuntime**)data);
  Deserializer derez(args, arglen);
  size_t length;
  derez.deserialize(length);
  std::string model_name((const char*)derez.get_current_pointer(), length);
  derez.advance_pointer(length);
  uint64_t model_version;
  derez.deserialize(model_version);
  unsigned instance_index;
  derez.deserialize(instance_index);
  size_t num_inputs;
  derez.deserialize(num_inputs);
  std::vector<InputTensor> inputs(num_inputs);
  for (unsigned idx1 = 0; idx1 < num_inputs; idx1++) {
    InputTensor& tensor = inputs[idx1];
    size_t namelen;
    derez.deserialize(namelen);
    tensor.name_ =
        std::string((const char*)derez.get_current_pointer(), namelen);
    derez.advance_pointer(namelen);
    size_t num_buffers;
    derez.deserialize(num_buffers);
    tensor.buffers_.resize(num_buffers);
    for (unsigned idx2 = 0; idx2 < num_buffers; idx2++)
      derez.deserialize(tensor.buffers_[idx2]);
    tensor.buffer_locations_.resize(num_buffers);
    for (unsigned idx2 = 0; idx2 < num_buffers; idx2++) {
      auto& pair = tensor.buffer_locations_[idx2];
      derez.deserialize(pair.first);
      derez.deserialize(pair.second);
    }
    tensor.buffer_memories_.resize(num_buffers);
    for (unsigned idx2 = 0; idx2 < num_buffers; idx2++)
      derez.deserialize(tensor.buffer_memories_[idx2]);
    size_t num_strides;
    derez.deserialize(num_strides);
    tensor.strides_.resize(num_strides);
    for (unsigned idx2 = 0; idx2 < num_strides; idx2++)
      derez.deserialize(tensor.strides_[idx2]);
  }
  size_t num_outputs;
  derez.deserialize(num_outputs);
  std::vector<OutputTensor> outputs(num_outputs);
  for (unsigned idx1 = 0; idx1 < num_outputs; idx1++) {
    OutputTensor& tensor = outputs[idx1];
    size_t namelen;
    derez.deserialize(namelen);
    tensor.name_ =
        std::string((const char*)derez.get_current_pointer(), namelen);
    derez.advance_pointer(namelen);
    size_t num_buffers;
    derez.deserialize(num_buffers);
    tensor.buffers_.resize(num_buffers);
    for (unsigned idx2 = 0; idx2 < num_buffers; idx2++)
      derez.deserialize(tensor.buffers_[idx2]);
    tensor.buffer_locations_.resize(num_buffers);
    for (unsigned idx2 = 0; idx2 < num_buffers; idx2++) {
      auto& pair = tensor.buffer_locations_[idx2];
      derez.deserialize(pair.first);
      derez.deserialize(pair.second);
    }
    tensor.buffer_memories_.resize(num_buffers);
    for (unsigned idx2 = 0; idx2 < num_buffers; idx2++)
      derez.deserialize(tensor.buffer_memories_[idx2]);
    size_t num_strides;
    derez.deserialize(num_strides);
    tensor.strides_.resize(num_strides);
    for (unsigned idx2 = 0; idx2 < num_strides; idx2++)
      derez.deserialize(tensor.strides_[idx2]);
  }
  Realm::Barrier barrier;
  derez.deserialize(barrier);
  Realm::UserEvent pretrigger, posttrigger;
  derez.deserialize(pretrigger);
  derez.deserialize(posttrigger);
  AddressSpaceID source;
  derez.deserialize(source);
  assert(!derez.get_remaining_bytes());

  std::vector<uint64_t> dummy_input_timing, dummy_output_timing;
  runtime->DistributeRunModel(
      model_name, model_version, instance_index, inputs, outputs,
      dummy_input_timing, dummy_output_timing, source,
      NULL /*unknown instance*/, barrier, pretrigger, posttrigger);
}

/*static*/ void
LegionTritonRuntime::InitCudaTask(
    const void* args, size_t arglen, const void* data, size_t datalen,
    Realm::Processor p)
{
#ifdef LEGION_USE_CUDA
  assert(arglen == sizeof(unsigned));
  const unsigned index = *((const unsigned*)args);
  assert(datalen == sizeof(LegionTritonRuntime*));
  LegionTritonRuntime* runtime = *((LegionTritonRuntime**)data);
  CHECK_CUDNN(cudnnCreate(&(runtime->cudnn[index])));
  CHECK_CUBLAS(cublasCreate(&(runtime->cublas[index])));
#else
  abort();
#endif
}

/*static*/ void
LegionTritonRuntime::LoadLayerTask(
    const void* args, size_t arglen, const void* data, size_t datalen,
    Realm::Processor p)
{
  assert(arglen == sizeof(Operator**));
  assert(datalen == sizeof(LegionTritonRuntime*));
  Operator* op = *((Operator**)args);
  op->Load(p);
}

/*static*/ void
LegionTritonRuntime::FreeLayerTask(
    const void* args, size_t arglen, const void* data, size_t datalen,
    Realm::Processor p)
{
  assert(arglen == sizeof(Operator**));
  assert(datalen == sizeof(LegionTritonRuntime*));
  Operator* op = *((Operator**)args);
  op->Free(p);
}

}}}  // namespace triton::backend::legion
