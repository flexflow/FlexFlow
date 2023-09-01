#include "runtime/legion_backing.h"
#include "kernels/nccl.h"
#include "model.h"
#include "runtime/task_spec/typed_task_invocation.h"
#include "task_spec/concrete_args_format.h"
#include "task_spec/device_specific.h"
#include "task_spec/future_args_format.h"
#include "task_spec/task_argument_accessor.h"
#include "task_spec/task_invocation_args_format.h"

using namespace Legion;
using namespace FlexFlow::Kernels;

namespace FlexFlow {

static void allocate_region_fields(LegionConfig const &config) {
  FieldAllocator allocator = config.runtime->create_field_allocator(
      config.context, config.field_space);
  allocator.allocate_field(sizeof(float), FID_DATA);
};

LegionBacking initialize_runtime(LegionConfig const &config) {
  allocate_region_fields(config);
  NOT_IMPLEMENTED();
}

IndexSpaceManager::IndexSpaceManager(LegionConfig const &config)
    : config(config), all_task_is() {}

/* IndexSpace const &IndexSpaceManager::at(MachineView const &view) const { */
/*   if (contains_key(this->all_task_is, view)) { */
/*     return all_task_is.at(view); */
/*   } */
/*   IndexSpace task_is; */
/*   Context ctx = config.lg_ctx; */
/*   Runtime *runtime = config.lg_hlr; */
/*   switch (view.num_dims()) { */
/* #define DIMFUNC(DIM) \ */
/*   case DIM: { \ */
/*     Rect<DIM> task_rect; \ */
/*     for (int i = 0; i < DIM; i++) { \ */
/*       task_rect.lo[i] = 0; \ */
/*       task_rect.hi[i] = view.at(i).num_points - 1; \ */
/*     } \ */
/*     task_is = runtime->create_index_space(ctx, task_rect); \ */
/*     break; \ */
/*   } */
/*     LEGION_FOREACH_N(DIMFUNC) */
/* #undef DIMFUNC */
/*     default: */
/*       assert(false); */
/*   } */
/*   all_task_is[view] = task_is; */
/*   return all_task_is.at(view); */
/* } */

/* static MachineView get_example_view(Domain const &domain) { */
/*   std::vector<StridedRectangleSide> sides; */
/*   for (int i = 0; i < domain.get_dim(); i++) { */
/*     int size = domain.hi()[i] - domain.lo()[i] + 1; */
/*     sides.push_back({ size, 1 }); */
/*   } */
/*   StridedRectangle rect = { 0, sides }; */
/*   MachineView view = { DeviceType::GPU, rect }; */
/*   return view; */
/* } */

/* IndexSpace const &IndexSpaceManager::at(Domain const &domain) const { */
/*   return this->at(get_example_view(domain)); */
/* } */

/* static MachineView singleton_view() { */
/*   return { */
/*     DeviceType::GPU, */
/*     { */
/*       0, */
/*       { */
/*         {1, 1} */
/*       } */
/*     } */
/*   }; */
/* } */

/* static int get_index_space_dimension(ParallelTensorDims const &dims) { */
/*   std::vector<int> parallel_idxs; */
/*   for (ParallelDim const &dim : dims) { */
/*     if (dim.parallel_idx >= 0) { */
/*       parallel_idxs.push_back(dim.parallel_idx); */
/*     } */
/*   } */

/*   for (int parallel_idx : parallel_idxs) { */
/*     assert (parallel_idx < parallel_idxs.size()); */
/*   } */

/*   return parallel_idxs.size(); */
/* } */

/* IndexSpace IndexSpaceManager::get_or_create_task_is(ParallelTensorDims const
 * &dims) { */
/*   int index_space_dimension = get_index_space_dimension(dims); */
/*   if (index_space_dimension == 0) { */
/*     return get_or_create_task_is(singleton_view()); */
/*   } */

/*   std::vector<optional<StridedRectangleSide>> sides(index_space_dimension,
 * nullopt); */

/*   for (ParallelDim const &dim : dims) { */
/*     if (dim.parallel_idx >= 0) { */
/*       sides.at(dim.parallel_idx) = {dim.degree, 1}; */
/*     } */
/*   } */

/*   StridedRectangle rect = { 0, value_all(sides) }; */
/*   MachineView view = { DeviceType::GPU, rect }; */
/*   return get_or_create_task_is(view); */
/* } */

/* IndexSpace IndexSpaceManager::get_task_is(MachineView const &view) const { */
/*   return all_task_is.at(view); */
/* } */

/* IndexSpace IndexSpaceManager::get_task_is(ParallelTensorDims const &dims)
 * const { */
/*   int index_space_dimension = get_index_space_dimension(dims); */
/*   if (index_space_dimension == 0) { */
/*     return get_task_is(singleton_view()); */
/*   } */

/*   std::vector<optional<StridedRectangleSide>> sides(index_space_dimension,
 * nullopt); */

/*   for (ParallelDim const &dim : dims) { */
/*     if (dim.parallel_idx >= 0) { */
/*       sides.at(dim.parallel_idx) = {dim.degree, 1}; */
/*     } */
/*   } */

/*   StridedRectangle rect = { 0, value_all(sides) }; */
/*   MachineView view = { DeviceType::GPU, rect }; */
/*   return get_task_is(view); */

/* } */

/* IndexSpace FFModel::get_task_is(ParallelConfig const &pc) const { */
/*   MachineView view; */
/*   view.ndims = pc.nDims; */
/*   for (int i = 0; i < view.ndims; i++) { */
/*     view.dim[i] = pc.dim[i]; */
/*   } */
/*   return get_task_is(view); */
/* } */

/* IndexSpace FFModel::get_or_create_task_is(ParallelConfig const &pc) { */
/*   MachineView view; */
/*   view.ndims = pc.nDims; */
/*   for (int i = 0; i < view.ndims; i++) { */
/*     view.dim[i] = pc.dim[i]; */
/*   } */
/*   return get_or_create_task_is(view); */
/* } */

int get_num_nodes() {
  return Realm::Machine::get_machine().get_address_space_count();
}

LegionConfig::LegionConfig()
    : context(Runtime::get_context()), runtime(Runtime::get_runtime()) {
  this->field_space = this->runtime->create_field_space(this->context);
}

enum Slots { NCCL_UNIQUE_ID, FF_INIT_INFO };

static ncclUniqueId generate_nccl_unique_id_task(
    Legion::Task const *task,
    std::vector<Legion::PhysicalRegion> const &regions,
    Legion::Context ctx,
    Legion::Runtime *runtime) {
  return NCCL::generate_unique_id();
}

static ncclComm_t
    init_nccl_comms_task(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  // Must be an index space launch
  assert(task->is_index_space);
  auto ncclId = acc.get_argument<ncclUniqueId>(NCCL_UNIQUE_ID);
  int allRanks = task->index_domain.get_volume();
  assert(task->index_domain.contains(task->index_point));
  int myRank = 0;
  for (Legion::Domain::DomainPointIterator it(task->index_domain); it;
       it++, myRank++) {
    if (it.p == task->index_point) {
      break;
    }
  }
  // fprintf(stderr, "ncclComm(%p) allRanks(%d) myRank(%d) ncclId(%p)\n",
  //     ncclComm, allRanks, myRank, ncclId);
  return NCCL::create_comm(ncclId, allRanks, myRank);
}

TypedStandardTaskInvocation<ncclUniqueId> get_unique_id() {
  return ensure_return_type<ncclUniqueId>(
      {NCCL_GETUNIQUEID_TASK_ID, StandardTaskBinding{}});
}

/* TypedTaskInvocation<ncclComm_t>
 * initialize_nccl_communicator(IndexSpaceManager const &idx_mgr, */
/*                                                              MachineView
 * const &machine_view) { */
/*   auto b = TaskBinding::index_launch(machine_view); */
/*   b.bind_arg(NCCL_UNIQUE_ID, get_unique_id()); */
/*   return ensure_return_type<ncclComm_t>({ NCCL_INIT_COMMS_TASK_ID, b }); */
/* } */

/* for (size_t l = 0; l < operators.size(); l++) { */
// Only create nccl for weights
/* if (operators[l]->op_type != OP_WEIGHT) { */
/*   continue; */
/* } */
/* MachineView view = operators[l]->outputs[0]->machine_view.value(); */
/* if (view_to_comms.find(get_std_hash(view)) == */
/*     view_to_comms.end()) { */

// TaskLauncher launcher(NCCL_GETUNIQUEID_TASK_ID, TaskArgument(NULL, 0));
// Future future = runtime->execute_task(ctx, launcher);
// ncclUniqueId ncclId = future.get_result<ncclUniqueId>();
// IndexSpace task_is = this->index_space_mgr.get_or_create_task_is(view);
// ArgumentMap argmap;
// IndexLauncher index_launcher(
//     NCCL_INIT_COMMS_TASK_ID,
//     task_is,
//     TaskArgument(&ncclId, sizeof(ncclUniqueId)),
//     argmap,
//     Predicate::TRUE_PRED,
//     false /*must*/,
//     0 /*mapper_id*/,
//     get_std_hash(view) /*MappingTagID*/);
// FutureMap fm = runtime->execute_index_space(ctx, index_launcher);
// fm.wait_all_results();
// int idx = 0;
// Domain task_domain = runtime->get_index_space_domain(ctx, task_is);
// ncclComm_t *nccl_comms =
//     (ncclComm_t *)malloc(sizeof(ncclComm_t) * task_domain.get_volume());
// for (Domain::DomainPointIterator it(task_domain); it; it++, idx++) {
//   nccl_comms[idx] = fm.get_result<ncclComm_t>(*it);
// }
// view_hash_to_nccl_comms[get_std_hash(view)] = nccl_comms;

ncclComm_t *NcclCommunicators::at(MachineView const &view) const {
  return this->view_to_comms.at(view);
}

static void ff_init_task(Legion::Task const *task,
                         std::vector<Legion::PhysicalRegion> const &regions,
                         Legion::Context ctx,
                         Legion::Runtime *runtime) {}

TensorlessIndexTaskInvocation ff_init(FFConfig const &config,
                                      FFInitInfo const &info) {
  MachineView mv = get_basic_data_parallel_machine_view(config);
  TensorlessIndexTaskBinding b(mv);
  b.bind_arg(FF_INIT_INFO, info);

  return {FF_INIT_TASK_ID, b};
}

template <>
void register_task<NCCL_GETUNIQUEID_TASK_ID>() {
  TaskSignature sig;
  sig.add_return_value<ncclUniqueId>();

  register_task(NCCL_GETUNIQUEID_TASK_ID,
                "NCCL GetUniqueId Task",
                sig,
                generate_nccl_unique_id_task);
}

template <>
void register_task<NCCL_INIT_COMMS_TASK_ID>() {
  TaskSignature sig;
  sig.add_arg_slot<ncclUniqueId>(NCCL_UNIQUE_ID);
  sig.add_return_value<ncclComm_t>();

  register_task(NCCL_INIT_COMMS_TASK_ID,
                "NCCL Init Communicators Task",
                sig,
                init_nccl_comms_task);
}

template <>
void register_task<FF_INIT_TASK_ID>() {
  TaskSignature sig;
  sig.add_arg_slot<FFInitInfo>(FF_INIT_INFO);
  sig.add_return_value<DeviceSpecific<PerDeviceFFHandle>>();

  register_task(FF_INIT_TASK_ID, "cuda init task", sig, ff_init_task);
}

} // namespace FlexFlow
