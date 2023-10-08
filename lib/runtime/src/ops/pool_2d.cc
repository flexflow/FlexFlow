#include "pool_2d.h"
#include "kernels/pool_2d_kernels.h"
#include "legion/legion_utilities.h"
#include "op-attrs/ops/pool_2d.h"
#include "utils/exception.decl.h"
#include "utils/exceptions.h"
#include "utils/hash-utils.h"
#include "op-attrs/get_output_shapes.h"

using namespace FlexFlow::Kernels::Pool2D;

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

Tensor FFModel::pool2d(const Tensor input,
                       int kernelH,
                       int kernelW,
                       int strideH,
                       int strideW,
                       int paddingH,
                       int paddingW,
                       PoolType type,
                       ActiMode activation,
                       char const *name) {
  assert(input->num_dims == 4); /*NCHW*/
  Layer *pool = new Layer(this,
                          OP_POOL2D,
                          DT_FLOAT,
                          name,
                          1 /*inputs*/,
                          0 /*weights*/,
                          1 /*outputs*/,
                          input);
  int numdims = 4;
  int dims[MAX_TENSOR_DIM];
  dims[3] = input->dims[3];
  dims[2] = input->dims[2];
  dims[1] = 1 + (input->dims[1] + 2 * paddingH - kernelH) / strideH;
  dims[0] = 1 + (input->dims[0] + 2 * paddingW - kernelW) / strideW;
  pool->outputs[0] = create_tensor_legion_ordering(
      numdims, dims, DT_FLOAT, pool, 0, true /*create_grad*/);
  pool->add_int_property("kernel_h", kernelH);
  pool->add_int_property("kernel_w", kernelW);
  pool->add_int_property("stride_h", strideH);
  pool->add_int_property("stride_w", strideW);
  pool->add_int_property("padding_h", paddingH);
  pool->add_int_property("padding_w", paddingW);
  pool->add_int_property("pool_type", type);
  pool->add_int_property("activation", activation);
  layers.push_back(pool);
  return pool->outputs[0];

#ifdef DEACODE
  Pool2D *pool = new Pool2D(*this,
                            input,
                            kernelH,
                            kernelW,
                            strideH,
                            strideW,
                            paddingH,
                            paddingW,
                            type,
                            activation,
                            name);
  layers.push_back(pool);
  return pool->outputs[0];
#endif
}

Op *Pool2D::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("kernel_h", value);
  int kernelH = value;
  layer->get_int_property("kernel_w", value);
  int kernelW = value;
  layer->get_int_property("stride_h", value);
  int strideH = value;
  layer->get_int_property("stride_w", value);
  int strideW = value;
  layer->get_int_property("padding_h", value);
  int paddingH = value;
  layer->get_int_property("padding_w", value);
  int paddingW = value;
  layer->get_int_property("pool_type", value);
  PoolType type = (PoolType)value;
  layer->get_int_property("activation", value);
  ActiMode activation = (ActiMode)value;
  return new Pool2D(model,
                    inputs[0],
                    kernelH,
                    kernelW,
                    strideH,
                    strideW,
                    paddingH,
                    paddingW,
                    type,
                    activation,
                    layer->name);
}

Pool2DParams Pool2D::get_params() const {
  Pool2DParams params;
  params.kernel_h = this->kernel_h;
  params.kernel_w = this->kernel_w;
  params.stride_h = this->stride_h;
  params.stride_w = this->stride_w;
  params.padding_h = this->padding_h;
  params.padding_w = this->padding_w;
  params.pool_type = this->pool_type;
  params.activation = this->activation;

  return params;
}

using PCG::Node;
bool operator==(Pool2DParams const &lhs, Pool2DParams const &rhs) {
  return lhs.kernel_h == rhs.kernel_h && lhs.kernel_w == rhs.kernel_w &&
         lhs.stride_h == rhs.stride_h && lhs.stride_w == rhs.stride_w &&
         lhs.padding_h == rhs.padding_h && lhs.padding_w == rhs.padding_w &&
         lhs.pool_type == rhs.pool_type && lhs.activation == rhs.activation;
}

int Pool2DParams::output_size(ParallelTensorShape const &input,
                              ParallelDim output_dims[MAX_TENSOR_DIM]) const {
  int input_w = input.dims[Pool2DInput::WIDTH].size;
  int input_h = input.dims[Pool2DInput::HEIGHT].size;
  int input_c = input.dims[Pool2DInput::CHANNEL].size;
  int input_n = input.dims[Pool2DInput::SAMPLE].size;

  output_dims[Pool2DOutput::WIDTH].size =
      1 + (input_w + 2 * padding_w - kernel_w) / stride_w;
  output_dims[Pool2DOutput::HEIGHT].size =
      1 + (input_h + 2 * padding_h - kernel_h) / stride_h;
  output_dims[Pool2DOutput::CHANNEL].size = input_c;
  output_dims[Pool2DOutput::SAMPLE].size = input_n;
  output_dims[Pool2DOutput::REPLICA].is_replica_dim = true;

  return Pool2DOutput::NUMDIM;
}

Pool2D::Pool2D(FFModel &model, Pool2D const &other, ParallelTensor const input)
    : Pool2D(model,
             input,
             other.kernel_h,
             other.kernel_w,
             other.stride_h,
             other.stride_w,
             other.padding_h,
             other.padding_w,
             other.pool_type,
             other.activation,
             other.name) {}

Pool2D::Pool2D(FFModel &model,
               const ParallelTensor _input,
               int _kernel_h,
               int _kernel_w,
               int _stride_h,
               int _stride_w,
               int _padding_h,
               int _padding_w,
               PoolType _type,
               ActiMode _activation,
               char const *name)
    : Op(model,
         OP_POOL2D,
         _input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         _input),
      kernel_h(_kernel_h), kernel_w(_kernel_w), stride_h(_stride_h),
      stride_w(_stride_w), padding_h(_padding_h), padding_w(_padding_w),
      pool_type(_type), activation(_activation) {
  assert(_input->num_dims == Pool2DInput::NUMDIM);

  Pool2D::construct_output_mappings(*this->parallel_dims_mapping);

  ParallelDim output_dims[MAX_TENSOR_DIM];
  int output_ndims;
  this->get_params().solve_dims(
      this->inputs[0]->get_shape(), output_dims, &output_ndims);

  outputs[0] = model.create_parallel_tensor_legion_ordering(
      output_ndims, output_dims, DT_FLOAT, this);
}

Pool2D::Pool2D(FFModel &model,
               Pool2DParams const &params,
               const ParallelTensor input,
               char const *name)
    : Pool2D(model,
             input,
             params.kernel_h,
             params.kernel_w,
             params.stride_h,
             params.stride_w,
             params.padding_h,
             params.padding_w,
             params.pool_type,
             params.activation,
             name) {}

void Pool2D::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher init_launcher(POOL2D_INIT_TASK_ID,
                              parallel_is,
                              TaskArgument(this, sizeof(Pool2D)),
                              argmap,
                              Predicate::TRUE_PRED,
                              false /*must*/,
                              0 /*mapper_id*/,
                              outputs[0]->machine_view.hash());
  init_launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                         0 /*projection id*/,
                                                         READ_ONLY,
                                                         EXCLUSIVE,
                                                         inputs[0]->region));
  init_launcher.add_field(0, FID_DATA);
  init_launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                         0 /*projection id*/,
                                                         WRITE_DISCARD,
                                                         EXCLUSIVE,
                                                         outputs[0]->region));
  init_launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, init_launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

/*
  regions[0]: input
  regions[1]: output
*/
PerDeviceOpState *Pool2D::init_task(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  Pool2D const *pool = (Pool2D *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  Pool2DMeta *m = new Pool2DMeta(handle);
  m->profiling = pool->profiling;
  std::strcpy(m->op_name, pool->name);
  TensorAccessorR<float, Pool2DInput::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Pool2DOutput::NUMDIM> acc_output(regions[1],
                                                          task->regions[1],
                                                          FID_DATA,
                                                          ctx,
                                                          runtime,
                                                          false /*readOutput*/);

  int input_w = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int input_h = acc_input.rect.hi[1] - acc_input.rect.lo[1] + 1;
  int input_c = acc_input.rect.hi[2] - acc_input.rect.lo[2] + 1;
  int input_n = acc_input.rect.hi[3] - acc_input.rect.lo[3] + 1;
  int output_w = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int output_h = acc_output.rect.hi[1] - acc_output.rect.lo[1] + 1;
  int output_c = acc_output.rect.hi[2] - acc_output.rect.lo[2] + 1;
  int output_n = acc_output.rect.hi[3] - acc_output.rect.lo[3] + 1;

  printf("init pool (input): n(%d) c(%d) h(%d) w(%d)\n",
         input_n,
         input_c,
         input_h,
         input_w);
  printf("init pool (output): n(%d) c(%d) h(%d) w(%d)\n",
         output_n,
         output_c,
         output_h,
         output_w);

  int pad_h =
      ((output_h - 1) * pool->stride_h + pool->kernel_h - input_h + 1) / 2;
  int pad_w =
      ((output_w - 1) * pool->stride_w + pool->kernel_w - input_w + 1) / 2;
  if (pad_h != pool->padding_h) {
    printf("Warning: changing pool_padding_h to satisfy output_h size\n");
  }
  if (pad_w != pool->padding_w) {
    printf("Warning: changing pool_padding_w to satisfy output_w size\n");
  }

  init_kernel(m,
              input_w,
              input_h,
              input_c,
              input_n,
              output_w,
              output_h,
              output_c,
              output_n,
              pad_h,
              pad_w,
              pool->kernel_h,
              pool->kernel_w,
              pool->stride_h,
              pool->stride_w,
              pool->pool_type);
  return m;
}

void Pool2D::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(POOL2D_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_DISCARD,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](O): output
*/
void Pool2D::forward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  // const Pool2D* pool = (Pool2D*) task->args;
  Pool2DMeta const *m = *((Pool2DMeta **)task->local_args);
  TensorAccessorR<float, Pool2DInput::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Pool2DOutput::NUMDIM> acc_output(regions[1],
                                                          task->regions[1],
                                                          FID_DATA,
                                                          ctx,
                                                          runtime,
                                                          false /*readOutput*/);

  forward_kernel_wrapper(m, acc_input.ptr, acc_output.ptr);
}

void Pool2D::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(POOL2D_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(NULL, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): input
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  // regions[1](I/O): input_grad
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  // regions[2](I): output
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(2, FID_DATA);
  // regions[3](I): output_grad
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(3, FID_DATA);

  runtime->execute_index_space(ctx, launcher);
}

/*
  regions[0](I): input
  regions[1](I/O): input_grad
  regions[2](I): output
  regions[3](I): output_grad
*/
void Pool2D::backward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(regions.size() == 4);
  assert(task->regions.size() == 4);
  // const Pool2D* pool = (Pool2D*) task->args;
  Pool2DMeta const *m = *((Pool2DMeta **)task->local_args);
  TensorAccessorR<float, Pool2DInput::NUMDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<float, Pool2DInput::NUMDIM> acc_input_grad(
      regions[1],
      task->regions[1],
      FID_DATA,
      ctx,
      runtime,
      true /*readOutput*/);
  TensorAccessorR<float, Pool2DOutput::NUMDIM> acc_output(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  TensorAccessorR<float, Pool2DOutput::NUMDIM> acc_output_grad(
      regions[3], task->regions[3], FID_DATA, ctx, runtime);

  backward_kernel_wrapper(m,
                          acc_input.ptr,
                          acc_input_grad.ptr,
                          acc_output.ptr,
                          acc_output_grad.ptr);
}

void Pool2D::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->kernel_h);
  sez.serialize(this->kernel_w);
  sez.serialize(this->stride_h);
  sez.serialize(this->stride_w);
  sez.serialize(this->padding_h);
  sez.serialize(this->padding_w);
  sez.serialize(this->pool_type);
  sez.serialize(this->activation);
}

bool Pool2D::measure_operator_cost(Simulator *sim,
                                   MachineView const &mv,
                                   CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output, sub_input;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  int input_w = sub_input.dims[0].size;
  int input_h = sub_input.dims[1].size;
  int input_c = sub_input.dims[2].size;
  int input_n = sub_input.dims[3].size;
  int output_w = sub_output.dims[0].size;
  int output_h = sub_output.dims[1].size;
  int output_c = sub_output.dims[2].size;
  int output_n = sub_output.dims[3].size;
  int pad_h = ((output_h - 1) * stride_h + kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * stride_w + kernel_w - input_w + 1) / 2;
  Pool2DMeta *m = sim->pool2d_meta;

  init_kernel(m,
              input_w,
              input_h,
              input_c,
              input_n,
              output_w,
              output_h,
              output_c,
              output_n,
              pad_h,
              pad_w,
              this->kernel_h,
              this->kernel_w,
              this->stride_h,
              this->stride_w,
              this->pool_type);
  // allocate tensors in simulator
  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  assert(m->profiling == false);

  std::function<void()> forward, backward;
  forward = [&] { forward_kernel_wrapper(m, input_ptr, output_ptr); };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr =
        (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert(input_grad_ptr != NULL);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    float *output_grad_ptr =
        (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert(output_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    backward = [&] {
      backward_kernel_wrapper(
          m, input_ptr, input_grad_ptr, output_ptr, output_grad_ptr);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure Pool2D] name(%s) input(%d %d %d %d) output(%d "
                      "%d %d %d) stride(%d %d) padding(%d %d) "
                      "forward_time(%.4lf) backward_time(%.4lf)\n",
                      name,
                      input_n,
                      input_c,
                      input_h,
                      input_w,
                      output_n,
                      output_c,
                      output_h,
                      output_w,
                      stride_h,
                      stride_w,
                      padding_h,
                      padding_w,
                      cost_metrics.forward_time,
                      cost_metrics.backward_time);
  } else {
    log_measure.debug(
        "[Measure Pool2D] name(%s) input(%d %d %d %d) output(%d %d %d %d) "
        "stride(%d %d) padding(%d %d) forward_time(%.4lf)\n",
        name,
        input_n,
        input_c,
        input_h,
        input_w,
        output_n,
        output_c,
        output_h,
        output_w,
        stride_h,
        stride_w,
        padding_h,
        padding_w,
        cost_metrics.forward_time);
  }

  return true;
}

using PCG::Node;
/*static*/
Node Pool2D::deserialize(FFModel &ff,
                         Legion::Deserializer &dez,
                         ParallelTensor inputs[],
                         int num_inputs) {
  assert(num_inputs == 1);

  int kernel_h, kernel_w, stride_h, stride_w, padding_h, padding_w;
  PoolType pool_type;
  ActiMode activation;

  dez.deserialize(kernel_h);
  dez.deserialize(kernel_w);
  dez.deserialize(stride_h);
  dez.deserialize(stride_w);
  dez.deserialize(padding_h);
  dez.deserialize(padding_w);
  dez.deserialize(pool_type);
  dez.deserialize(activation);

  Pool2DParams params;
  params.kernel_h = kernel_h;
  params.kernel_w = kernel_w;
  params.stride_h = stride_h;
  params.stride_w = stride_w;
  params.padding_h = padding_h;
  params.padding_w = padding_w;
  params.pool_type = pool_type;
  params.activation = activation;

  return ff.get_or_create_node<Pool2D>(inputs[0], params);
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::Pool2DParams>::operator()(
    FlexFlow::Pool2DParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.kernel_h);
  hash_combine(key, params.kernel_w);
  hash_combine(key, params.stride_h);
  hash_combine(key, params.stride_w);
  hash_combine(key, params.padding_h);
  hash_combine(key, params.padding_w);
  hash_combine(key, params.pool_type);
  hash_combine(key, params.activation);
  return key;
}

enum Slots { INPUT, OUTPUT, ATTRS, PROFILING, PER_DEVICE_STATE, HANDLE};

OpTaskInvocation init(Pool2DAttrs const & attrs) {
  OpTaskBinding binding;
  binding.bind(INPUT, input_tensor(0));
  binding.bind(OUTPUT, output_tensor(0));
  binding.bind_arg(ATTRS, attrs);
  binding.bind_arg(HANDLE,ff_handle());

  return {POOL2D_INIT_TASK_ID, binding};
}

static DeviceSpecific<Pool2dPerDeviceState> init_task_impl(TaskArgumentAccessor const &acc) {
  NOT_IMPLEMENTED();
  auto const &attrs = acc.get_argument<Pool2DAttrs>(ATTRS);
  PerDeviceFFHandle handle = acc.get_argument<PerDeviceFFHandle>(HANDLE);

  auto input = acc.get_tensor<Permission::RO>(INPUT);
  auto output = acc.get_tensor<Permission::WO>(OUTPUT);

  int input_w = input.shape.at(ff_dim_t(0)) + 1
  int input_h = input.shape.at(ff_dim_t(1)) + 1
  int input_c = input.shape.at(ff_dim_t(2)) + 1
  int input_n = input.shape.at(ff_dim_t(3)) + 1
  int output_w = output.shape.at(ff_dim_t(0)) + 1
  int output_h = output.shape.at(ff_dim_t(1)) + 1
  int output_c = output.shape.at(ff_dim_t(2)) + 1
  int output_n = output.shape.at(ff_dim_t(3)) + 1

  printf("init pool (input): n(%d) c(%d) h(%d) w(%d)\n",
         input_n,
         input_c,
         input_h,
         input_w);
  printf("init pool (output): n(%d) c(%d) h(%d) w(%d)\n",
         output_n,
         output_c,
         output_h,
         output_w);

  int pad_h = ((output_h - 1) * attrs.stride_h + attrs.kernel_h - input_h + 1) / 2;
  int pad_w = ((output_w - 1) * attrs.stride_w + attrs.kernel_w - input_w + 1) / 2;
  if (pad_h != attrs.padding_h) {
    printf("Warning: changing pool_padding_h to satisfy output_h size\n");
  }

  if (pad_w != attrs.padding_w) {
    printf("Warning: changing pool_padding_w to satisfy output_w size\n");
  }

  DeviceSpecific<Pool2dPerDeviceState> state = acc.create_device_specific<Pool2dPerDeviceState>(
              init_kernel(handle,
                          attrs.activation,
                          input_w,
                          input_h,
                          input_c,
                          input_n,
                          output_w,
                          output_h,
                          output_c,
                          output_n,
                          pad_h,
                          pad_w,
                          attrs.kernel_h,
                          attrs.kernel_w,
                          attrs.stride_h,
                          attrs.stride_w,
                          attrs.pool_type);

  return state;
}

static DeviceSpecific<Pool2dPerDeviceState>  init_task(Task const *task,
              std::vector<PhysicalRegion> const &regions,
              Context ctx,
              Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  return init_task_impl(acc);
}

OpTaskInvocation forward(Pool2DAttrs const & attrs) {
    OpTaskBinding binding;
    binding.bind(INPUT, input_tensor(0));
    binding.bind(OUTPUT, output_tensor(0));

    binding.bind_arg(PROFILING, profiling_settings());
    binding.bind_arg(PER_DEVICE_STATE, per_device_op_state<Pool2dPerDeviceState>());

    return {POOL2D_FWD_TASK_ID, binding};
}

OpTaskInvocation backward(Pool2DAttrs const &) {
  OpTaskBinding b = infer_bwd_binding(forward(attrs).binding);

  return {POOL2D_BWD_TASK_ID, b};
}

static optional<float> forward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  Pool2dPerDeviceState state = acc.get_argument<Pool2dPerDeviceState>(PER_DEVICE_STATE);

  auto input = acc.get_tensor<Permission::RO>(INPUT);
  auto output = acc.get_tensor<Permission::WO>(OUTPUT);

  return profile(forward_kernel,
                  profilng,
                  "[Pool2D] forward_time = %.2lfms\n",
                  state,
                  input.get_float_ptr(),
                  output.get_float_ptr());
}

static void forward_task(Task const *task,
                         std::vector<PhysicalRegion> const &regions,
                         Context ctx,
                         Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  forward_task_impl(acc);
}

static optional<float> backward_task_impl(TaskArgumentAccessor const &acc) {
  ProfilingSettings profiling = acc.get_argument<ProfilingSettings>(PROFILING);
  Pool2dPerDeviceState state = acc.get_argument<Pool2dPerDeviceState>(PER_DEVICE_STATE);

  auto input = acc.get_tensor<Permission::RO>(INPUT);
  auto input_grad = acc.get_tensor<Permission::RW>(INPUT);
  auto output = acc.get_tensor<Permission::RO>(OUTPUT);
  auto output_grad = acc.get_tensor<Permission::RO>(OUTPUT);

  return profile(backward_kernel,
                  profilng,
                  "[Pool2D] backward_time = %.2lfms\n",
                  state,
                  input.get_float_ptr(),
                  input_grad.get_float_ptr(),
                  output.get_float_ptr(),
                  output_grad.get_float_ptr());

}

static void backward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  TaskArgumentAccessor acc(task, regions, ctx, runtime);
  backward_task_impl(acc);
}

CostMetrics measure_operator_cost(SimEnvFactory const &sim_factory,
                                  Pool2DAttrs const &attrs,
                                  ParallelTensorShape const &input_shape,
                                  ProfilingSettings const &settings,
                                  MachineView const &machine_view) {
  auto env = sim.new_environment();
  ParallelTensorShape output_shape =
      get_output_shape(attrs, input_shape);
  
  SimTaskBinding init_binding;
  init_binding.bind(INPUT, input_shape);
  init_binding.bind(OUTPUT, output_shape);
  init_binding.bind_arg(ATTRS, attrs);
  init_binding.bind_arg(HANDLE, ff_handle());

  auto init_accessor =
      env.get_init_accessor(POOL2D_INIT_TASK_ID, init_binding);
  
  DeviceSpecific<Pool2dPerDeviceState> per_device_state = init_task_impl(init_accessor);

}

template <>
void register_task<POOL2D_INIT_TASK_ID>() {
  OpTaskSignature init(OpTaskType::INIT);

  init.add_input_slot(INPUT);
  init.add_output_slot(OUTPUT);

  init.add_arg_slot<Pool2DAttrs>(ATTRS);
  init.add_unchecked_arg_slot<PerDeviceFFHandle>(HANDLE);

  init.add_return_value<FlexFlow::Pool2DPerDeviceState>();

  register_task(POOL2D_INIT_TASK_ID, "Pool2D::init", init, init_taks); 
}

template <>
void register_task<POOL2D_FWD_TASK_ID>() {

}

template <>
void register_task<POOL2D_BWD_TASK_ID>() {

}


}; // namespace std
