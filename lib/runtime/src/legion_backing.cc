#include "legion_backing.h"

namespace FlexFlow {

#ifdef FF_USE_NCCL
static ncclUniqueId get_nccl_unique_id_task(Legion::Task const *task,
                                            std::vector<Legion::PhysicalRegion> const &regions,
                                            Legion::Context ctx,
                                            Legion::Runtime *runtime) {
  ncclUniqueId ncclId;
  checkNCCL(ncclGetUniqueId(&ncclId));
  return ncclId;
}

static ncclComm_t init_nccl_comms_task(Legion::Task const *task,
                                       std::vector<Legion::PhysicalRegion> const &regions,
                                       Legion::Context ctx,
                                       Legion::Runtime *runtime) {
  // Must be an index space launch
  assert(task->is_index_space);
  ncclUniqueId ncclId = *((ncclUniqueId const *)task->args);
  int allRanks = task->index_domain.get_volume();
  assert(task->index_domain.contains(task->index_point));
  int myRank = 0;
  for (Legion::Domain::DomainPointIterator it(task->index_domain); it; it++, myRank++) {
    if (it.p == task->index_point) {
      break;
    }
  }
  ncclComm_t ncclComm;
  checkNCCL(ncclCommInitRank(&ncclComm, allRanks, ncclId, myRank));
  // fprintf(stderr, "ncclComm(%p) allRanks(%d) myRank(%d) ncclId(%p)\n",
  //     ncclComm, allRanks, myRank, ncclId);
  return ncclComm;
}
#endif

TaskInvocation initialize_nccl_communicators(LegionConfig const &config, MachineView const &) {
  // init all nccl communicators
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  for (size_t l = 0; l < operators.size(); l++) {
    // Only create nccl for weights
    if (operators[l]->op_type != OP_WEIGHT) {
      continue;
    }
    MachineView view = operators[l]->outputs[0]->machine_view.value();
    if (view_to_comms.find(get_std_hash(view)) ==
        view_to_comms.end()) {
      TaskLauncher launcher(NCCL_GETUNIQUEID_TASK_ID, TaskArgument(NULL, 0));
      Future future = runtime->execute_task(ctx, launcher);
      ncclUniqueId ncclId = future.get_result<ncclUniqueId>();
      IndexSpace task_is = this->index_space_mgr.get_or_create_task_is(view);
      ArgumentMap argmap;
      IndexLauncher index_launcher(
          NCCL_INIT_COMMS_TASK_ID,
          task_is,
          TaskArgument(&ncclId, sizeof(ncclUniqueId)),
          argmap,
          Predicate::TRUE_PRED,
          false /*must*/,
          0 /*mapper_id*/,
          get_std_hash(view) /*MappingTagID*/);
      FutureMap fm = runtime->execute_index_space(ctx, index_launcher);
      fm.wait_all_results();
      int idx = 0;
      Domain task_domain = runtime->get_index_space_domain(ctx, task_is);
      ncclComm_t *nccl_comms =
          (ncclComm_t *)malloc(sizeof(ncclComm_t) * task_domain.get_volume());
      for (Domain::DomainPointIterator it(task_domain); it; it++, idx++) {
        nccl_comms[idx] = fm.get_result<ncclComm_t>(*it);
      }
      view_hash_to_nccl_comms[get_std_hash(view)] = nccl_comms;
    }
  }
}

#ifdef FF_USE_NCCL
ncclComm_t *FFModel::find_nccl_comms(MachineView const &view) const {
  size_t key = get_std_hash(view);
  if (contains_key(this->view_hash_to_nccl_comms, key)) {
    return this->view_hash_to_nccl_comms.at(key);
  } else {
    assert (config.computationMode == COMP_MODE_INFERENCE);
    return nullptr;
  }
}
#endif

enum Slots {
  NCCL_UNIQUE_ID
};

template <>
void register_task<NCCL_GETUNIQUEID_TASK_ID>() {
  TaskSignature sig;
  sig.add_return_value<ncclUniqueId>();

  register_task(NCCL_GETUNIQUEID_TASK_ID, "NCCL GetUniqueId Task", sig, get_nccl_unique_id_task);
}

template <>
void register_task<NCCL_INIT_COMMS_TASK_ID>() {
  TaskSignature sig;
  sig.add_arg_slot<ncclUniqueId>(NCCL_UNIQUE_ID);
  sig.add_return_value<ncclComm_t>();

  register_task(NCCL_INIT_COMMS_TASK_ID, "NCCL Init Communicators Task", sig, init_nccl_comms_task);
}


}
