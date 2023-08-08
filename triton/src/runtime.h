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

#ifndef __LEGION_TRITON_RUNTIME_H__
#define __LEGION_TRITON_RUNTIME_H__

#include "config.h"
#include "legion.h"
#include "triton/backend/backend_common.h"
#include "types.h"
#ifdef LEGION_USE_CUDA
#include "cudahelp.h"
#endif

namespace triton { namespace backend { namespace legion {

//
// Legion Triton Runtime
// This is a small runtime built to facilitate the glue operations
// between requests coming in from Triton and running inference jobs
// in Legion. It provides facilities for hooking up model instances
// across the differant processes so that they are all aligned even
// if models are loaded out of order. It also helps order requests
// to the different model instances coming from different nodes.
// Despite its name, its actually a Realm runtime since it works
// in parallel with Legion on top of Realm.
//
class LegionTritonRuntime {
 public:
  enum {
    // TODO: This is a bit of a hack in that we are "guessing" about
    // where Legion is going to register task variants with Realm
    // utility processors. In practice, Legion would need to register
    // 1M task variants with utility processors before it conflicted
    // with these task IDs. Legion registers at most a few task variants
    // with utility processors so conflicts should never happen in practice.
    // It would be nice to do this in a way that doesn't rely on having
    // knowledge of Legion's internals though.
    INSTANCE_CREATE_TASK_ID = 1 << 20,
    CONTEXT_CREATE_TASK_ID,
    RUN_MODEL_INFERENCE_TASK_ID,
    INIT_CUDALIBS_TASK_ID,
    LOAD_LAYER_TASK_ID,
    FREE_LAYER_TASK_ID,
  };
  struct PendingContext {
   public:
    PendingContext(const std::string& name, uint64_t v, unsigned idx)
        : model_name(name), version(v), index(idx)
    {
    }
    inline bool matches(const std::string& name, uint64_t v, unsigned idx) const
    {
      if (name != model_name)
        return false;
      if (version != v)
        return false;
      if (index != idx)
        return false;
      return true;
    }

   public:
    std::string model_name;
    uint64_t version;
    unsigned index;
    std::map<
        Realm::Processor, std::pair<LegionModelInstance*, Realm::UserEvent> >
        requests;
  };

 public:
  static TRITONSERVER_Error* Create(
      Legion::TaskID ttid, LegionTritonRuntime** runtime);

 private:
  LegionTritonRuntime(
      Legion::Runtime* lg, Legion::TaskID ttid, Legion::AddressSpaceID rank,
      size_t total, Realm::Processor local,
      std::vector<Realm::Processor>&& remote,
      std::vector<Realm::Processor>&& cpus,
      std::vector<Realm::Processor>&& gpus,
      const std::vector<Realm::Processor>& local_gpus,
      bool allowTensorOpMathConversion = true);

 public:
  void RecordModel(LegionModelState* model);
  void RemoveModel(LegionModelState* model);
  void RendezvousContextCreation(
      LegionModelInstance* instance, Realm::UserEvent ready);
  void DistributeRunModel(
      const std::string& model_name, uint64_t model_version,
      unsigned instance_index, const std::vector<InputTensor>& inputs,
      const std::vector<OutputTensor>& outputs,
      std::vector<uint64_t>& compute_input_end_ns,
      std::vector<uint64_t>& compute_output_start_ns,
      Legion::AddressSpaceID source, LegionModelInstance* instance = NULL,
      Realm::Barrier barrier = Realm::Barrier::NO_BARRIER,
      Realm::UserEvent pretrigger = Realm::UserEvent::NO_USER_EVENT,
      Realm::UserEvent posttrigger = Realm::UserEvent::NO_USER_EVENT);

 public:
  unsigned FindGPUIndex(Realm::Processor proc) const;
  const std::vector<Realm::Processor>& FindAllProcessors(
      Realm::Processor::Kind kind);
  const std::vector<Realm::Processor>& FindLocalProcessors(
      Realm::Processor::Kind kind);
  Realm::Memory FindMemory(TRITONSERVER_MemoryType type, uint64_t device_id);
  Realm::Event LoadLayer(Realm::Processor proc, Operator* op);
  Realm::Event FreeLayer(Realm::Processor proc, Operator* op);

 protected:
  void HandleContextCreation(
      const std::string& name, uint64_t version, unsigned index,
      Realm::Processor source, LegionModelInstance* instance,
      Realm::UserEvent ready, bool external = true, bool need_lock = true);
  void CreatePendingContext(const PendingContext& pending);
  Legion::AddressSpaceID InstanceOwner(unsigned instance_index) const;
  LegionModelInstance* FindModelInstance(
      const std::string& model_name, uint64_t model_version,
      unsigned instance_index, bool external, bool need_lock = true);

 public:
  static void InstanceMessageTask(
      const void* args, size_t arglen, const void* userdata, size_t userlen,
      Realm::Processor p);
  static void CreateContextTask(
      const void* args, size_t arglen, const void* userdata, size_t userlen,
      Realm::Processor p);
  static void RunModelInferenceTask(
      const void* args, size_t arglen, const void* userdata, size_t userlen,
      Realm::Processor p);
  static void InitCudaTask(
      const void* args, size_t arglen, const void* userdata, size_t userlen,
      Realm::Processor p);
  static void LoadLayerTask(
      const void* args, size_t arglen, const void* userdata, size_t userlen,
      Realm::Processor p);
  static void FreeLayerTask(
      const void* args, size_t arglen, const void* userdata, size_t userlen,
      Realm::Processor p);

 protected:
  static std::vector<Realm::Processor> FindLocal(
      Legion::AddressSpaceID local, const std::vector<Realm::Processor>& procs);
  static Realm::Memory FindLocalSysmem(void);
  static Realm::Memory FindLocalRegmem(void);
  static std::vector<Realm::Memory> FindLocalFramebuffers(
      const std::vector<Realm::Processor>& gpus);
  static void PackRunModel(
      Legion::Serializer& rez, const std::string& model_name,
      uint64_t model_version, unsigned instance_index,
      const std::vector<InputTensor>& inputs,
      const std::vector<OutputTensor>& outputs);

 public:
  Legion::Runtime* const legion_;
  const Legion::TaskID top_task_id_;
  const Legion::AddressSpaceID rank_;
  const size_t total_ranks_;
  const Realm::Processor local_proc_;
  const Realm::Memory local_sysmem_;
  const Realm::Memory local_regmem_;
  const std::vector<Realm::Processor> remote_procs_;
  const std::vector<Realm::Processor> all_cpus_;
  const std::vector<Realm::Processor> all_gpus_;
  const std::vector<Realm::Processor> local_cpus_;
  const std::vector<Realm::Processor> local_gpus_;
  const std::vector<Realm::Memory> local_framebuffers_;
  const bool allowTensorOpMathConversion_;
#ifdef LEGION_USE_CUDA
 public:
  cudnnHandle_t cudnn[MAX_LOCAL_PROCS];
  cublasHandle_t cublas[MAX_LOCAL_PROCS];
#endif
 private:
  Realm::FastReservation lock_;
  std::vector<LegionModelState*> models_;

 private:
  std::list<PendingContext> pending_contexts_;
  Realm::Barrier creation_barrier_;
  std::map<Realm::Processor, unsigned> gpu_lookup_;
};

// A small helper class for using Realm's fast reservation
// synchronization primitive in external threads
template <bool EXTERNAL>
class AutoLock {
 public:
  AutoLock(Realm::FastReservation& lock, bool exclusive = true) : lock_(lock)
  {
    const Realm::Event wait_on = exclusive ? lock_.wrlock() : lock_.rdlock();
    if (wait_on.exists() && !wait_on.has_triggered()) {
      if (EXTERNAL)
        wait_on.external_wait();
      else
        wait_on.wait();
    }
  }
  ~AutoLock(void) { lock_.unlock(); }

 private:
  Realm::FastReservation& lock_;
};

}}}  // namespace triton::backend::legion

#endif  // __LEGION_TRITON_RUNTIME_H__
