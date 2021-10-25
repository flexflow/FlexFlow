#include "flexflow/ops/dropout.h"
#include "legion/legion_utilities.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {
  
// declare Legion names
using Legion::Context;
using Legion::Runtime;
using Legion::TaskLauncher;
using Legion::IndexLauncher;
using Legion::FutureMap;
using Legion::ArgumentMap;
using Legion::TaskArgument;
using Legion::RegionRequirement;
using Legion::Predicate;
using Legion::InlineLauncher;

Tensor FFModel::dropout(const Tensor input,
                        float rate,
                        unsigned long long seed,
                        const char* name)
{
  Layer* dropout = new Layer(this, OP_DROPOUT, name, 1/*inputs*/,
                             0/*weights*/, 1/*outputs*/, input);
  int numdims = input->num_dims;
  int dims[MAX_TENSOR_DIM];
  for (int i = 0; i < numdims; i++)
    dims[i] = input->dims[i];
  dropout->outputs[0] = create_tensor_legion_ordering(numdims, dims, DT_FLOAT,
                                                      dropout, 0, true/*create_grad*/);
  dropout->add_float_property("rate", rate);
  dropout->add_int_property("seed", seed);
  layers.push_back(dropout);
  return dropout->outputs[0];
#ifdef DEADCODE
  // see = 0 is preserved as None, so we use a random seed
  if (seed == 0) {
    seed = std::rand();
  }
  Dropout *dropout = new Dropout(*this, input, rate, seed, name);
  layers.push_back(dropout);
  return dropout->outputs[0];
#endif
}

Dropout::Dropout(FFModel& model,
                 const ParallelTensor _input,
                 float _rate,
                 unsigned long long _seed,
                 const char* name)
: Op(model, OP_DROPOUT, name, 1/*inputs*/, 0/*weights*/, 1/*outputs*/, _input),
  rate(_rate), seed(_seed)
{
  // Set output shape
  ParallelDim dims[MAX_TENSOR_DIM];
  for (int i = 0; i < _input->num_dims; i++)
    dims[i] = _input->dims[i];
  numOutputs = 1;
  outputs[0] = model.create_parallel_tensor_legion_ordering(_input->num_dims, dims, DT_FLOAT, this);
}

#ifdef DEADCODE
void Dropout::map_output_tensors(FFModel& model)
{
  int dim = inputs[0].num_dims;
  switch (dim) {
#define DIMFUNC(DIM) \
    case DIM: \
    { \
      task_is = model.get_or_create_task_is(DIM, name); \
      map_output_tensors_with_dim<DIM>(model); \
      break; \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
    {
      assert(false && "Unsupported dim");
    }
  }
}

template<int NDIM>
void Dropout::map_output_tensors_with_dim(FFModel& model)
{
  // Retrive the task indexspace for the op
  task_is = IndexSpaceT<NDIM>(model.get_or_create_task_is(NDIM, name));
  Context ctx = model.config.lg_ctx;
  Runtime* runtime = model.config.lg_hlr;
  Rect<NDIM> part_rect = runtime->get_index_space_domain(ctx, task_is);
  int dims[NDIM];
  for (int i = 0; i < NDIM; i++)
    dims[i] = inputs[0].adim[NDIM-1-i];
  outputs[0] = model.create_tensor<NDIM>(dims, DT_FLOAT, this);
  outputs[0].owner_op = this;
  outputs[0].owner_idx = 0;
  Rect<NDIM> input_rect;
  input_rect = runtime->get_index_partition_color_space(
        ctx, inputs[0]->part.get_index_partition());
  if (input_rect == part_rect) {
    input_lps[0] = inputs[0]->part;
    input_grad_lps[0] = inputs[0]->part_grad;
  } else {
    model.create_disjoint_partition<NDIM>(
        inputs[0], IndexSpaceT<NDIM>(task_is), input_lps[0], input_grad_lps[0]);
  }
}
#endif

void Dropout::init(const FFModel& ff)
{
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher init_launcher(DROPOUT_INIT_TASK_ID, parallel_is,
                              TaskArgument(this, sizeof(Dropout)), argmap,
                              Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                              outputs[0]->machine_view.hash());
  init_launcher.add_region_requirement(
      RegionRequirement(inputs[0]->part, 0/*projection id*/,
                        READ_ONLY, EXCLUSIVE, inputs[0]->region));
  init_launcher.add_field(0, FID_DATA);
  init_launcher.add_region_requirement(
      RegionRequirement(outputs[0]->part, 0/*projection id*/,
                        WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  init_launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void Dropout::forward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(DROPOUT_FWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part, 0/*projection id*/,
      WRITE_ONLY, EXCLUSIVE, outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Dropout::backward(const FFModel& ff)
{
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(DROPOUT_BWD_TASK_ID, parallel_is,
                         TaskArgument(NULL, 0), argmap,
                         Predicate::TRUE_PRED, false/*must*/, 0/*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(
    RegionRequirement(inputs[0]->part_grad, 0/*projection id*/,
      READ_WRITE, EXCLUSIVE, inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
    RegionRequirement(outputs[0]->part_grad, 0/*projection id*/,
      READ_ONLY, EXCLUSIVE, outputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}


}; // namespace FlexFlow
