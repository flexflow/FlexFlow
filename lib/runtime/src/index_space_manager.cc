#include "index_space_manager.h"

using namespace Legion;

namespace FlexFlow {

IndexSpaceManager::IndexSpaceManager(LegionConfig const &config)
  : config(config), all_task_is()
{ }

IndexSpace const &IndexSpaceManager::at(MachineView const &view) const {
  if (contains_key(this->all_task_is, view)) {
    return all_task_is.at(view);
  }
  IndexSpace task_is;
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  switch (view.num_dims()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    Rect<DIM> task_rect;                                                       \
    for (int i = 0; i < DIM; i++) {                                            \
      task_rect.lo[i] = 0;                                                     \
      task_rect.hi[i] = view.at(i).num_points - 1;                             \
    }                                                                          \
    task_is = runtime->create_index_space(ctx, task_rect);                     \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  all_task_is[view] = task_is;
  return all_task_is.at(view);
}

static MachineView get_example_view(Domain const &domain) {
  std::vector<StridedRectangleSide> sides;
  for (int i = 0; i < domain.get_dim(); i++) {
    int size = domain.hi()[i] - domain.lo()[i] + 1;
    sides.push_back({ size, 1 });
  }
  StridedRectangle rect = { 0, sides };
  MachineView view = { DeviceType::GPU, rect };
  return view;
}

IndexSpace const &IndexSpaceManager::at(Domain const &domain) const {
  return this->at(get_example_view(domain));
}

static MachineView singleton_view() {
  return {
    DeviceType::GPU,
    {
      0, 
      {
        {1, 1}
      }
    }
  };
}

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


/* IndexSpace IndexSpaceManager::get_or_create_task_is(ParallelTensorDims const &dims) { */
/*   int index_space_dimension = get_index_space_dimension(dims); */
/*   if (index_space_dimension == 0) { */
/*     return get_or_create_task_is(singleton_view()); */
/*   } */

/*   std::vector<optional<StridedRectangleSide>> sides(index_space_dimension, nullopt); */

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

/* IndexSpace IndexSpaceManager::get_task_is(ParallelTensorDims const &dims) const { */
/*   int index_space_dimension = get_index_space_dimension(dims); */
/*   if (index_space_dimension == 0) { */
/*     return get_task_is(singleton_view()); */
/*   } */

/*   std::vector<optional<StridedRectangleSide>> sides(index_space_dimension, nullopt); */

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


}
