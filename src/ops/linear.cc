#include "flexflow/ops/linear.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/layer.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/linear_kernels.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {

// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::Future;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::InlineLauncher;
using Legion::Machine;
using Legion::Memory;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

using namespace FlexFlow::Kernels::Linear;

static constexpr int KERNEL_IDX = 0;
static constexpr int BIAS_IDX = 1;

Tensor FFModel::dense(const Tensor input,
                      int outDim,
                      ActiMode activation,
                      bool use_bias,
                      DataType data_type,
                      Layer const *shared_op,
                      Initializer *kernel_initializer,
                      Initializer *bias_initializer,
                      RegularizerMode kernel_reg_type,
                      float kernel_reg_lambda,
                      char const *name) {
  if (data_type == DT_NONE) {
    data_type = input->data_type;
  }
  DataType quantization_type = cpu_offload ? config.quantization_type : DT_NONE;
  bool offload = cpu_offload;
  Layer *li = nullptr;
  if (data_type != input->data_type) {
    Tensor casted_input = cast(input, data_type, "type cast for dense");
    li = new Layer(this,
                   OP_LINEAR,
                   data_type,
                   name,
                   1 /*inputs*/,
                   use_bias ? 2 : 1 /*weights*/,
                   1 /*outputs*/,
                   casted_input);
  } else {
    li = new Layer(this,
                   OP_LINEAR,
                   data_type,
                   name,
                   1 /*inputs*/,
                   use_bias ? 2 : 1 /*weights*/,
                   1 /*outputs*/,
                   input);
  }

  {
    int numdims = input->num_dims;
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++) {
      dims[i] = input->dims[i];
    }
    dims[0] = outDim;
    li->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, data_type, li, 0, true /*create_grad*/);
  }
  {
    int dims[2] = {input->dims[0], outDim};
    if (quantization_type != DT_NONE) {
      dims[0] =
          get_quantization_to_byte_size(data_type, quantization_type, dims[0]);
    }
    li->weights[KERNEL_IDX] = create_weight_legion_ordering(
        2,
        dims,
        quantization_type == DT_NONE ? data_type : quantization_type,
        li,
        true /*create_grad*/,
        kernel_initializer,
        CHOSEN_SYNC_TYPE);
  }
  if (use_bias) {
    int dims[1] = {outDim};
    li->weights[BIAS_IDX] = create_weight_legion_ordering(1,
                                                          dims,
                                                          data_type,
                                                          li,
                                                          true /*create_grad*/,
                                                          bias_initializer,
                                                          CHOSEN_SYNC_TYPE);
  }
  li->add_int_property("use_bias", use_bias);
  li->add_int_property("out_dim", outDim);
  li->add_int_property("activation", activation);
  li->add_int_property("kernel_reg_type", kernel_reg_type);
  li->add_float_property("kernel_reg_lambda", kernel_reg_lambda);
  li->add_int_property("quantization_type", quantization_type);
  li->add_int_property("offload", offload);
  layers.push_back(li);
  return li->outputs[0];
}

Op *Linear::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  long long value;
  layer->get_int_property("use_bias", value);
  bool use_bias = (bool)value;
  layer->get_int_property("out_dim", value);
  int outdim = value;
  layer->get_int_property("activation", value);
  ActiMode activation = (ActiMode)value;
  layer->get_int_property("kernel_reg_type", value);
  RegularizerMode kernel_reg_type = (RegularizerMode)value;
  float kernel_reg_lambda;
  layer->get_float_property("kernel_reg_lambda", kernel_reg_lambda);
  layer->get_int_property("quantization_type", value);
  DataType quantization_type = (DataType)value;
  layer->get_int_property("offload", value);
  bool offload = (bool)value;
  return new Linear(model,
                    layer->layer_guid,
                    inputs[0],
                    outdim,
                    activation,
                    kernel_reg_type,
                    kernel_reg_lambda,
                    use_bias,
                    layer->data_type,
                    quantization_type,
                    offload,
                    false /*allocate_weights*/,
                    layer->name);
}

// size_t Linear::get_params_hash() const {
//   return this->get_params().get_hash(this->inputs[0]);
// }

Linear::Linear(FFModel &model,
               Linear const &other,
               const ParallelTensor input,
               bool allocate_weights)
    : Linear(model,
             other.layer_guid,
             input,
             other.out_channels,
             other.activation,
             other.kernel_reg_type,
             other.kernel_reg_lambda,
             other.use_bias,
             other.data_type,
             other.quantization_type,
             other.offload,
             allocate_weights,
             other.name) {}

Linear::Linear(FFModel &model,
               LinearParams const &params,
               ParallelTensor const input,
               char const *name,
               bool allocate_weights)
    : Linear(model,
             params.layer_guid,
             input,
             params.out_channels,
             params.activation,
             params.kernel_reg_type,
             params.kernel_reg_lambda,
             params.use_bias,
             params.data_type,
             params.quantization_type,
             params.offload,
             allocate_weights,
             name) {}

Linear::Linear(FFModel &model,
               LayerID const &_layer_guid,
               const ParallelTensor _input,
               int out_dim,
               ActiMode _activation,
               RegularizerMode _kernel_reg_type,
               float _kernel_reg_lambda,
               bool _use_bias,
               DataType _data_type,
               DataType _quantization_type,
               bool _offload,
               bool allocate_weights,
               char const *name)
    : Op(model,
         OP_LINEAR,
         _data_type,
         name,
         1 /*inputs*/,
         _use_bias ? 2 : 1 /*weights*/,
         allocate_weights,
         1 /*outputs*/,
         _input),
      out_channels(out_dim), activation(_activation), use_bias(_use_bias),
      kernel_reg_type(_kernel_reg_type), kernel_reg_lambda(_kernel_reg_lambda),
      quantization_type(_quantization_type), offload(_offload),
      replica(ParallelTensorBase::NO_TENSOR) {
  // overwrite layer_guid
  layer_guid = _layer_guid;
  data_type = _data_type;
  auto dimension_names =
      this->get_params().get_dimension_names(_input->get_shape());
  this->in_channels =
      _input->dims[dimension_names.at(LinearParams::INPUT_CHANNEL)].size;

  ParallelTensorShape input_shape = this->inputs[0]->get_shape();
  ParallelTensorShape output_shape, kernel_shape, bias_shape;
  LinearParams params = this->get_params();
  params.construct_mappings(*this->parallel_dims_mapping, input_shape);
  params.solve_dims(input_shape, output_shape, kernel_shape, bias_shape);
  kernel_shape.dims[0].size = this->in_channels;
  bias_shape.dims[0].degree = _input->dims[_input->num_dims - 1].degree;
  bias_shape.dims[0].parallel_idx =
      _input->dims[_input->num_dims - 1].parallel_idx;
  bias_shape.dims[1].size = bias_shape.dims[1].degree = 1;
  bias_shape.dims[1].parallel_idx = -1;
  bias_shape.dims[bias_shape.num_dims - 1].size =
      bias_shape.dims[bias_shape.num_dims - 1].degree = 1;
  for (int i = 0; i < input_shape.num_dims - 1; i++) {
    if (_input->dims[i].degree > 1) {
      bias_shape.dims[bias_shape.num_dims - 1].size *= _input->dims[i].degree;
      bias_shape.dims[bias_shape.num_dims - 1].degree *= _input->dims[i].degree;
      bias_shape.dims[bias_shape.num_dims - 1].parallel_idx =
          _input->dims[i].parallel_idx;
    }
  }

  if (allocate_weights) {
    Initializer *kernel_initializer = new GlorotUniform(std::rand() /*seed*/);
    if (quantization_type != DT_NONE) {
      kernel_shape.dims[0].size = get_quantization_to_byte_size(
          data_type, quantization_type, kernel_shape.dims[0].size);
    }
    weights[KERNEL_IDX] = model.create_parallel_weight_legion_ordering(
        kernel_shape.num_dims,
        kernel_shape.dims,
        quantization_type == DT_NONE ? _data_type : quantization_type,
        NULL /*owner_op*/,
        true /*create_grad*/,
        kernel_initializer,
        CHOSEN_SYNC_TYPE);

    if (use_bias) {
      Initializer *bias_initializer = new ZeroInitializer();

      weights[BIAS_IDX] =
          model.create_parallel_weight_legion_ordering(bias_shape.num_dims,
                                                       bias_shape.dims,
                                                       _data_type,
                                                       NULL /*owner_op*/,
                                                       true /*create_grad*/,
                                                       bias_initializer,
                                                       CHOSEN_SYNC_TYPE);
      add_bias_only_once = _input->dims[0].degree > 1;
    }
  }

  // Create the output tensor
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      output_shape.num_dims, output_shape.dims, _data_type, this);

  // assert(check_output_input_weight_parallel_dims(allocate_weights));
}

void Linear::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  // assert(check_output_input_weight_same_machine_view());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(LINEAR_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Linear)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // launcher.add_region_requirement(
  //     RegionRequirement(input_lps[0], 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, inputs[0]->region));
  // launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  // launcher.add_region_requirement(
  //     RegionRequirement(weights[1]->part, 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, weights[1]->region));
  // launcher.add_field(3, FID_DATA);
  if (ff.config.computationMode == COMP_MODE_TRAINING) {
    // Add inputs[0].region_grad to avoid Legion warning
    // launcher.add_region_requirement(
    //    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
    //        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
    // launcher.add_field(2, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
}

void Linear::init_inference(FFModel const &ff,
                            std::vector<ParallelTensor> const &batch_inputs,
                            std::vector<ParallelTensor> const &batch_outputs,
                            MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  // assert(check_output_input_weight_same_machine_view());
  parallel_is = batch_outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_init_inference(ff, argmap, batch_outputs[0]);
  IndexLauncher launcher(LINEAR_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Linear)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  // launcher.add_region_requirement(
  //     RegionRequirement(input_lps[0], 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, inputs[0]->region));
  // launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part,
                        0 /*projection id*/,
                        READ_ONLY,
                        EXCLUSIVE,
                        weights[0]->region,
                        ff.cpu_offload ? MAP_TO_ZC_MEMORY : 0));
  launcher.add_field(2, FID_DATA);
  // launcher.add_region_requirement(
  //     RegionRequirement(weights[1]->part, 0/*projection id*/,
  //                       READ_ONLY, EXCLUSIVE, weights[1]->region));
  // launcher.add_field(3, FID_DATA);
  if (ff.config.computationMode == COMP_MODE_TRAINING) {
    // Add inputs[0].region_grad to avoid Legion warning
    // launcher.add_region_requirement(
    //    RegionRequirement(input_grad_lps[0], 0/*projection id*/,
    //        WRITE_ONLY, EXCLUSIVE, inputs[0].region_grad));
    // launcher.add_field(2, FID_DATA);
  }
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, batch_outputs[0]);
}

/*
  regions[0](O): output
  regions[1](I): kernel
  regions[2](I): bias
*/
OpMeta *Linear::init_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  Linear const *linear = (Linear *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  GenericTensorAccessorW output =
      helperGetGenericTensorAccessorWO(linear->inputs[0]->data_type,
                                       regions[0],
                                       task->regions[0],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  switch (output.domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    if (output.data_type == DT_HALF) {                                         \
      if (linear->quantization_type != DT_NONE) {                              \
        return init_task_with_dim<half, char, DIM>(                            \
            task, regions, ctx, runtime);                                      \
      } else {                                                                 \
        return init_task_with_dim<half, half, DIM>(                            \
            task, regions, ctx, runtime);                                      \
      }                                                                        \
    } else if (output.data_type == DT_FLOAT) {                                 \
      if (linear->quantization_type != DT_NONE) {                              \
        return init_task_with_dim<float, char, DIM>(                           \
            task, regions, ctx, runtime);                                      \
      } else {                                                                 \
        return init_task_with_dim<float, float, DIM>(                          \
            task, regions, ctx, runtime);                                      \
      }                                                                        \
    } else {                                                                   \
      assert(false && "Unsupported data type");                                \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  return NULL;
}

template <typename DT, typename WT, int NDIM>
OpMeta *Linear::init_task_with_dim(Task const *task,
                                   std::vector<PhysicalRegion> const &regions,
                                   Context ctx,
                                   Runtime *runtime) {
  assert(regions.size() == task->regions.size());
  assert(regions.size() == 2 || regions.size() == 3);
  Linear const *linear = (Linear *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  // TensorAccessorR<float, 2> acc_input(
  //     regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<DT, NDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<DT, NDIM> acc_output(regions[1],
                                       task->regions[1],
                                       FID_DATA,
                                       ctx,
                                       runtime,
                                       false /*readOutput*/);
  TensorAccessorW<WT, NDIM> acc_kernel(regions[2],
                                       task->regions[2],
                                       FID_DATA,
                                       ctx,
                                       runtime,
                                       false /*readOutput*/);

  // TensorAccessorR<float, 1> acc_bias(
  //     regions[3], task->regions[3], FID_DATA, ctx, runtime);
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  // int in_dim = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  printf("init linear (input): in_dim(%d) out_dim(%d) batch_size(%d)\n",
         in_dim,
         out_dim,
         batch_size);
  Memory gpu_mem = Machine::MemoryQuery(Machine::get_machine())
                       .only_kind(Memory::GPU_FB_MEM)
                       .best_affinity_to(task->target_proc)
                       .first();
  MemoryAllocator gpu_mem_allocator(gpu_mem);
  if (linear->offload) {
    // cpu-offload enabled
    // use offload_reserved_space
    gpu_mem_allocator.register_reserved_work_space(
        handle.offload_reserve_space, handle.offload_reserve_space_size);
  }

  LinearMeta *m = new LinearMeta(
      handle, batch_size, linear, gpu_mem_allocator, in_dim * out_dim);
  m->activation = linear->activation;
  m->kernel_reg_type = linear->kernel_reg_type;
  m->kernel_reg_lambda = linear->kernel_reg_lambda;
  m->use_bias = linear->use_bias;
  m->add_bias_only_once = linear->add_bias_only_once;
  m->profiling = linear->profiling;
  m->trainableInputs[0] = linear->trainableInputs[0];
  m->weight_ptr_type = m->input_type[0];
  m->quantization_type = linear->quantization_type;
  m->offload = linear->offload;

  m->findBestAlgoID(out_dim, batch_size, in_dim);
  if (m->use_bias) {
    m->findBestAlgoID(out_dim, batch_size, 1);
  }
  std::strcpy(m->op_name, linear->name);

  init_kernel(m, batch_size, out_dim);

  return m;
}

void Linear::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(LINEAR_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
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
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    weights[0]->region));
  launcher.add_field(2, FID_DATA);
  if (use_bias) {
    launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[1]->region));
    launcher.add_field(3, FID_DATA);
  }
  runtime->execute_index_space(ctx, launcher);
}

FutureMap Linear::inference(FFModel const &ff,
                            BatchConfigFuture const &bc,
                            std::vector<ParallelTensor> const &batch_inputs,
                            std::vector<ParallelTensor> const &batch_outputs,
                            MachineView const *mv) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  parallel_is = batch_outputs[0]->parallel_is;
  MachineView const *view = mv ? mv : &batch_outputs[0]->machine_view;
  set_argumentmap_for_inference(ff, argmap, batch_outputs[0]);
  size_t machine_view_hash = view->hash();
  /* std::cout << "Linear op machine_view: " << *(MachineView const *)mv
            << std::endl; */
  IndexLauncher launcher(LINEAR_INF_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_outputs[0]->part,
                                                    0 /*projection id*/,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    batch_outputs[0]->region));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(weights[0]->part,
                        0 /*projection id*/,
                        READ_ONLY,
                        EXCLUSIVE,
                        weights[0]->region,
                        ff.cpu_offload ? MAP_TO_ZC_MEMORY : 0));
  launcher.add_field(2, FID_DATA);
  if (use_bias) {
    launcher.add_region_requirement(RegionRequirement(weights[1]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[1]->region));
    launcher.add_field(3, FID_DATA);
  }
  return runtime->execute_index_space(ctx, launcher);
}

void Linear::inference_task(Task const *task,
                            std::vector<PhysicalRegion> const &regions,
                            Context ctx,
                            Runtime *runtime) {
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  LinearMeta const *m = *((LinearMeta **)task->local_args);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  assert(regions.size() == (3 + static_cast<size_t>(m->use_bias)));
  assert(task->regions.size() == (3 + static_cast<size_t>(m->use_bias)));
  if (m->quantization_type == DT_NONE) {
    assert(m->input_type[0] == m->weight_type[0]);
  }
  assert(m->input_type[0] == m->output_type[0]);

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  GenericTensorAccessorR weight = helperGetGenericTensorAccessorRO(
      m->weight_type[0], regions[2], task->regions[2], FID_DATA, ctx, runtime);
  int in_dim = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  int out_dim = output.domain.hi()[0] - output.domain.lo()[0] + 1;

  int batch_size = bc->num_active_tokens();
  GenericTensorAccessorR bias;
  if (m->use_bias &&
      !(m->add_bias_only_once && task->index_point.point_data[0] != 0)) {
    bias = helperGetGenericTensorAccessorRO(m->weight_type[1],
                                            regions[3],
                                            task->regions[3],
                                            FID_DATA,
                                            ctx,
                                            runtime);
    assert(bias.domain.get_volume() == static_cast<size_t>(out_dim));
  }
  forward_kernel_wrapper(m,
                         input.ptr,
                         output.ptr,
                         weight.ptr,
                         bias.ptr,
                         in_dim,
                         out_dim,
                         batch_size);
}

void Linear::forward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  LinearMeta const *m = *((LinearMeta **)task->local_args);
  if (m->quantization_type == DT_NONE) {
    assert(m->input_type[0] == m->weight_type[0]);
  }
  assert(m->input_type[0] == m->output_type[0]);
  switch (input_domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    if (m->output_type[0] == DT_HALF) {                                        \
      if (m->quantization_type != DT_NONE) {                                   \
        return forward_task_with_dim<half, char, DIM>(                         \
            task, regions, ctx, runtime);                                      \
      } else {                                                                 \
        return forward_task_with_dim<half, half, DIM>(                         \
            task, regions, ctx, runtime);                                      \
      }                                                                        \
    } else if (m->output_type[0] == DT_FLOAT) {                                \
      if (m->quantization_type != DT_NONE) {                                   \
        return forward_task_with_dim<float, char, DIM>(                        \
            task, regions, ctx, runtime);                                      \
      } else {                                                                 \
        return forward_task_with_dim<float, float, DIM>(                       \
            task, regions, ctx, runtime);                                      \
      }                                                                        \
    } else {                                                                   \
      assert(false && "Unsupported data type");                                \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

/*
  regions[0](I); input
  regions[1](O): output
  regions[2](I): kernel
  regions[3](I): bias
*/
template <typename DT, typename WT, int NDIM>
void Linear::forward_task_with_dim(Task const *task,
                                   std::vector<PhysicalRegion> const &regions,
                                   Context ctx,
                                   Runtime *runtime) {
  // Linear* linear = (Linear*) task->args;
  LinearMeta const *m = *((LinearMeta **)task->local_args);
  assert(regions.size() == (3 + static_cast<size_t>(m->use_bias)));
  assert(task->regions.size() == (3 + static_cast<size_t>(m->use_bias)));

  TensorAccessorR<DT, NDIM> acc_input(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorW<DT, NDIM> acc_output(regions[1],
                                       task->regions[1],
                                       FID_DATA,
                                       ctx,
                                       runtime,
                                       false /*readOutput*/);
  TensorAccessorR<WT, NDIM> acc_kernel(
      regions[2], task->regions[2], FID_DATA, ctx, runtime);
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  assert(acc_output.rect.volume() == static_cast<size_t>(out_dim * batch_size));
  assert(acc_input.rect.volume() == static_cast<size_t>(in_dim * batch_size));
  // assert(acc_kernel.rect.volume() == static_cast<size_t>(in_dim * out_dim));
  DT const *acc_bias_ptr = nullptr;
  if (m->use_bias &&
      !(m->add_bias_only_once && task->index_point.point_data[0] != 0)) {
    TensorAccessorR<DT, NDIM> acc_bias(
        regions[3], task->regions[3], FID_DATA, ctx, runtime);
    assert(acc_bias.rect.volume() == static_cast<size_t>(out_dim));
    acc_bias_ptr = acc_bias.ptr;
  }

  forward_kernel_wrapper(m,
                         acc_input.ptr,
                         acc_output.ptr,
                         acc_kernel.ptr,
                         acc_bias_ptr,
                         in_dim,
                         out_dim,
                         batch_size);
}

void Linear::backward(FFModel const &ff) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  {
    ArgumentMap argmap;
    set_argumentmap_for_backward(ff, argmap);
    IndexLauncher launcher(LINEAR_BWD_TASK_ID,
                           parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           outputs[0]->machine_view.hash());
    int rid = 0;
    // regions[0](I): input
    launcher.add_region_requirement(RegionRequirement(inputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      inputs[0]->region));
    launcher.add_field(rid++, FID_DATA);
    // regions[1](I/O): replica_grad
    assert(replica == NULL);
    if (trainableInputs[0]) {
      launcher.add_region_requirement(
          RegionRequirement(inputs[0]->part_grad,
                            0 /*projection id*/,
                            READ_WRITE,
                            EXCLUSIVE,
                            inputs[0]->region_grad));
      launcher.add_field(rid++, FID_DATA);
    }
    // regions[2](I): output
    launcher.add_region_requirement(RegionRequirement(outputs[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      outputs[0]->region));
    launcher.add_field(rid++, FID_DATA);
    // regions[3](I/O): output_grad
    launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      outputs[0]->region_grad));
    launcher.add_field(rid++, FID_DATA);
    // regions[4](I): filter
    launcher.add_region_requirement(RegionRequirement(weights[0]->part,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      weights[0]->region));
    launcher.add_field(rid++, FID_DATA);
    // regions[5](I/O): filter_grad
    launcher.add_region_requirement(RegionRequirement(weights[0]->part_grad,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      weights[0]->region_grad));
    launcher.add_field(rid++, FID_DATA);
    if (use_bias) {
      // regions[6](I/O): bias_grad
      launcher.add_region_requirement(
          RegionRequirement(weights[1]->part_grad,
                            0 /*projection id*/,
                            READ_WRITE,
                            EXCLUSIVE,
                            weights[1]->region_grad));
      launcher.add_field(rid++, FID_DATA);
    }
    runtime->execute_index_space(ctx, launcher);
  }
  assert(replica == NULL);
}

void Linear::backward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  Domain in_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  LinearMeta const *m = *((LinearMeta **)task->local_args);
  if (m->quantization_type == DT_NONE) {
    assert(m->input_type[0] == m->weight_type[0]);
  }
  assert(m->input_type[0] == m->output_type[0]);
  switch (in_domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    if (m->output_type[0] == DT_HALF) {                                        \
      return backward_task_with_dim<half, DIM>(task, regions, ctx, runtime);   \
    } else if (m->output_type[0] == DT_FLOAT) {                                \
      return backward_task_with_dim<float, DIM>(task, regions, ctx, runtime);  \
    } else {                                                                   \
      assert(false && "Unsupported data type");                                \
    }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
}

/*
  regions[0](I): input
  regions[1](I/O): replica_grad or input_grad
  regions[2](I): output
  regions[3](I/O): output_grad
  regions[4](I): filter
  regions[5](I/O): filter_grad
  regions[6](I/O): bias_grad
*/
template <typename DT, int NDIM>
void Linear::backward_task_with_dim(Task const *task,
                                    std::vector<PhysicalRegion> const &regions,
                                    Context ctx,
                                    Runtime *runtime) {
  // Linear* linear = (Linear*) task->args;
  LinearMeta const *m = *((LinearMeta **)task->local_args);
  assert(regions.size() == (5 + static_cast<size_t>(m->trainableInputs[0]) +
                            static_cast<size_t>(m->use_bias)));
  assert(task->regions.size() ==
         (5 + static_cast<size_t>(m->trainableInputs[0]) +
          static_cast<size_t>(m->use_bias)));
  DT *input_grad = nullptr;
  size_t rid = 0;
  TensorAccessorR<DT, NDIM> acc_input(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  if (m->trainableInputs[0]) {
    Domain domain = runtime->get_index_space_domain(
        ctx, task->regions[rid].region.get_index_space());
    if (domain.get_dim() == NDIM + 1) {
      assert(domain.get_volume() == acc_input.rect.volume());
      input_grad = helperGetTensorPointerWO<DT>(
          regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
    } else {
      TensorAccessorW<DT, NDIM> acc_replica_grad(regions[rid],
                                                 task->regions[rid],
                                                 FID_DATA,
                                                 ctx,
                                                 runtime,
                                                 true /*readOutput*/);
      assert(acc_replica_grad.rect.volume() == acc_input.rect.volume());
      input_grad = acc_replica_grad.ptr;
    }
    rid++;
  }
  TensorAccessorR<DT, NDIM> acc_output(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  TensorAccessorW<DT, NDIM> acc_output_grad(regions[rid],
                                            task->regions[rid],
                                            FID_DATA,
                                            ctx,
                                            runtime,
                                            true /*readOutput*/);
  rid++;
  TensorAccessorR<DT, NDIM> acc_kernel(
      regions[rid], task->regions[rid], FID_DATA, ctx, runtime);
  rid++;
  TensorAccessorW<DT, NDIM> acc_kernel_grad(regions[rid],
                                            task->regions[rid],
                                            FID_DATA,
                                            ctx,
                                            runtime,
                                            true /*readOutput*/);
  rid++;
  // make sure the sizes match
  int in_dim = acc_input.rect.hi[0] - acc_input.rect.lo[0] + 1;
  int out_dim = acc_output.rect.hi[0] - acc_output.rect.lo[0] + 1;
  int batch_size = acc_output.rect.volume() / out_dim;
  assert(acc_output.rect.volume() == static_cast<size_t>(out_dim * batch_size));
  assert(acc_output_grad.rect.volume() ==
         static_cast<size_t>(out_dim * batch_size));
  assert(acc_kernel.rect.volume() == static_cast<size_t>(in_dim * out_dim));
  assert(acc_kernel_grad.rect.volume() ==
         static_cast<size_t>(in_dim * out_dim));
  DT *acc_bias_grad_ptr = nullptr;
  if (m->use_bias) {
    TensorAccessorW<DT, 3> acc_bias_grad(regions[rid],
                                         task->regions[rid],
                                         FID_DATA,
                                         ctx,
                                         runtime,
                                         true /*readOutput*/);
    rid++;
    assert(acc_bias_grad.rect.volume() == static_cast<size_t>(out_dim));
    acc_bias_grad_ptr = static_cast<DT *>(acc_bias_grad.ptr);
  }
  assert(rid == regions.size());

  backward_kernel_wrapper(m,
                          acc_input.ptr,
                          input_grad,
                          acc_output.ptr,
                          acc_output_grad.ptr,
                          acc_kernel.ptr,
                          acc_kernel_grad.ptr,
                          acc_bias_grad_ptr,
                          in_dim,
                          out_dim,
                          batch_size);
}

void Linear::print_layer(FFModel const &ff) {
  printf("linear layer\n");
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;

  RegionRequirement kernel_req(
      weights[0]->region, READ_WRITE, EXCLUSIVE, weights[0]->region);
  kernel_req.add_field(FID_DATA);
  InlineLauncher kernel_launcher(kernel_req);
  PhysicalRegion kernel_region = runtime->map_region(ctx, kernel_launcher);
  kernel_region.wait_until_valid();

  RegionRequirement bias_req(
      weights[1]->region, READ_WRITE, EXCLUSIVE, weights[1]->region);
  bias_req.add_field(FID_DATA);
  InlineLauncher bias_launcher(bias_req);
  PhysicalRegion bias_region = runtime->map_region(ctx, bias_launcher);
  bias_region.wait_until_valid();

  TensorAccessorW<float, 2> acc_kernel(
      kernel_region, kernel_req, FID_DATA, ctx, runtime, true);
  TensorAccessorW<float, 1> acc_bias(
      bias_region, bias_req, FID_DATA, ctx, runtime, true);

  float const *kernel_ptr = acc_kernel.ptr;
  float const *bias_ptr = acc_bias.ptr;

  size_t kernel_size = acc_kernel.rect.volume();
  int kernel_dim1 = acc_kernel.rect.hi[0] - acc_kernel.rect.lo[0] + 1;
  int kernel_dim2 = acc_kernel.rect.hi[1] - acc_kernel.rect.lo[1] + 1;
  size_t bias_size = acc_bias.rect.volume();
  printf("kernel, %p, %zu, [%d, %d]\n",
         kernel_ptr,
         kernel_size,
         kernel_dim1,
         kernel_dim2);
  printf("bias, %p, %zu\n", bias_ptr, bias_size);

  for (size_t i = 0; i < bias_size; i++) {
    printf("%f ", bias_ptr[i]);
  }
  printf("\n");

  for (size_t i = 0; i < kernel_size; i++) {
    printf("%f ", kernel_ptr[i]);
  }
  printf("\n");

  runtime->unmap_region(ctx, kernel_region);
  runtime->unmap_region(ctx, bias_region);
}

bool Linear::estimate_sync_cost(Simulator *sim,
                                MachineView const &view,
                                CostMetrics &cost_metrics) const {
  // Estimate the cost of sync weights
  ParallelTensorShape tensor_shape;
  tensor_shape.num_dims = 3;
  tensor_shape.data_type = inputs[0]->data_type;
  tensor_shape.dims[0] = inputs[0]->dims[0];
  tensor_shape.dims[1] = inputs[0]->dims[inputs[0]->num_dims - 1];
  tensor_shape.dims[2] = inputs[0]->dims[inputs[0]->num_dims - 2];
  tensor_shape.dims[1].size = out_channels;
  tensor_shape.dims[1].degree = 1;
  tensor_shape.dims[2].degree =
      inputs[0]->dims[1].degree * inputs[0]->dims[2].degree;
  tensor_shape.dims[2].size =
      inputs[0]->dims[1].degree * inputs[0]->dims[2].degree;
  cost_metrics.sync_time =
      sim->default_estimate_sync_cost(tensor_shape, view, 1);
  // printf("[Estimate Linear] name(%s) sync_time(%.4lf)\n", name,
  // cost_metrics.sync_time);
  return true;
}

ParallelConfig Linear::get_random_parallel_config(FFModel const &ff) const {
  if (!ff.config.enable_parameter_parallel) {
    return Op::get_random_parallel_config(ff);
  }
  std::vector<int> batch_candidates;
  std::vector<int> channel_candidates;
  int batch = outputs[0]->dims[outputs[0]->num_dims - 1].size;
  int channel = outputs[0]->dims[0].size;
  int total_devices = ff.config.workersPerNode * ff.config.numNodes;
  for (int i = 1; i <= ff.config.workersPerNode; i++) {
    if (channel % i == 0) {
      for (int j = 1; i * j <= total_devices; j++) {
        if (batch % j == 0) {
          batch_candidates.push_back(j);
          channel_candidates.push_back(i);
        }
      }
    }
  }
  assert(batch_candidates.size() > 0);
  int idx = std::rand() % batch_candidates.size();
  int num_par_c = channel_candidates[idx];
  int num_par_b = batch_candidates[idx];
  ParallelConfig pc;
  pc.device_type = ParallelConfig::GPU;
  pc.nDims = outputs[0]->num_dims;
  pc.dim[0] = num_par_c;
  pc.dim[pc.nDims - 1] = num_par_b;
  for (int i = 1; i < pc.nDims - 1; i++) {
    pc.dim[i] = 1;
  }
  int start_idx = std::rand() % (total_devices - num_par_c * num_par_b + 1);
  start_idx = start_idx - start_idx % num_par_c;
  for (int i = 0; i < num_par_c * num_par_b; i++) {
    pc.device_ids[i] = start_idx + i;
  }
  return pc;
}

bool Linear::get_int_parameter(PMParameter para, int *value) const {
  switch (para) {
    case PM_ACTI:
      *value = (int)activation;
      return true;
    default:
      return Op::get_int_parameter(para, value);
  }
}

bool Linear::is_valid_parallel_config(FFModel const &ff,
                                      ParallelConfig const &pc) const {
  if (!ff.config.enable_parameter_parallel) {
    return Op::is_valid_parallel_config(ff, pc);
  }
  // Support data and parameter parallel
  if (pc.nDims != outputs[0]->num_dims) {
    return false;
  }
  for (int i = 1; i < pc.nDims - 1; i++) {
    if (pc.dim[i] != 1) {
      return false;
    }
  }
  return true;
}

bool Linear::measure_operator_cost(Simulator *sim,
                                   MachineView const &mv,
                                   CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_output, sub_input;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  int input_c = sub_input.dims[0].size;
  int input_n = sub_input.get_volume() / input_c;
  int output_c = sub_output.dims[0].size;
  int output_n = sub_output.get_volume() / output_c;
  LinearMeta *m = sim->linear_meta;
  m->activation = activation;
  m->kernel_reg_type = kernel_reg_type;
  m->kernel_reg_lambda = kernel_reg_lambda;
  m->input_type[0] = inputs[0]->data_type;
  m->weight_type[0] = this->data_type;
  m->output_type[0] = outputs[0]->data_type;
  assert(m->profiling == false);

  init_kernel(m, output_n, output_c);

  // allocate tensors in simulator
  sim->free_all();
  void *input_ptr = sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  void *output_ptr =
      sim->allocate(sub_output.get_volume(), outputs[0]->data_type);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

  void *kernel_ptr = sim->allocate((size_t)output_c * input_c, this->data_type);
  void *bias_ptr = sim->allocate(output_c, this->data_type);
  assert(bias_ptr != NULL);
  cost_metrics.weights_memory += cost_metrics.total_mem_diff_from(sim->offset);

  bool out_of_memory = (input_ptr == NULL) || (output_ptr == NULL) ||
                       (kernel_ptr == NULL) || (bias_ptr == NULL);
  if (out_of_memory) {
    cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
    return true;
  }
  std::function<void()> forward, backward;
  forward = [&] {
    forward_kernel_wrapper(m,
                           input_ptr,
                           output_ptr,
                           kernel_ptr,
                           bias_ptr,
                           input_c,
                           output_c,
                           input_n);
  };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    void *input_grad_ptr = NULL;
    if (trainableInputs[0]) {
      input_grad_ptr =
          sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
    } else {
      input_grad_ptr =
          sim->allocate(sub_input.get_volume(), inputs[0]->data_type);
    }
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);

    void *output_grad_ptr =
        sim->allocate(sub_output.get_volume(), outputs[0]->data_type);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    void *kernel_grad_ptr =
        sim->allocate((size_t)output_c * input_c, this->data_type);
    void *bias_grad_ptr = sim->allocate(output_c, this->data_type);
    cost_metrics.weights_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);

    out_of_memory = (input_grad_ptr == NULL) || (output_grad_ptr == NULL) ||
                    (kernel_grad_ptr == NULL) || (bias_grad_ptr == NULL);
    if (out_of_memory) {
      cost_metrics.forward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      cost_metrics.backward_time = Simulator::MAXIMUM_TASK_RUN_TIME;
      return true;
    }
    backward = [&] {
      backward_kernel_wrapper(m,
                              input_ptr,
                              input_grad_ptr,
                              output_ptr,
                              output_grad_ptr,
                              kernel_ptr,
                              kernel_grad_ptr,
                              bias_grad_ptr,
                              input_c,
                              output_c,
                              input_n);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    log_measure.debug("[Measure Linear] name(%s) in(%d %d) out(%d %d) "
                      "forward_time(%.4lf) backward_time(%.4lf)\n",
                      name,
                      input_n,
                      input_c,
                      output_n,
                      output_c,
                      cost_metrics.forward_time,
                      cost_metrics.backward_time);
  } else {
    log_measure.debug(
        "[Measure Linear] name(%s) in(%d %d) out(%d %d) forward_time(%.4lf)\n",
        name,
        input_n,
        input_c,
        output_n,
        output_c,
        cost_metrics.forward_time);
  }
  return true;
}

bool operator==(LinearParams const &lhs, LinearParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid &&
         lhs.out_channels == rhs.out_channels && lhs.use_bias == rhs.use_bias &&
         lhs.data_type == rhs.data_type && lhs.activation == rhs.activation &&
         lhs.kernel_reg_type == rhs.kernel_reg_type &&
         lhs.kernel_reg_lambda == rhs.kernel_reg_lambda;
}

void Linear::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->out_channels);
  sez.serialize(this->activation);
  sez.serialize(this->kernel_reg_type);
  sez.serialize(this->kernel_reg_lambda);
  sez.serialize(this->use_bias);
  sez.serialize(this->data_type);
  sez.serialize(this->quantization_type);
  sez.serialize(this->offload);
}

/* static */
using PCG::Node;
Node Linear::deserialize(FFModel &ff,
                         Legion::Deserializer &dez,
                         ParallelTensor inputs[],
                         int num_inputs) {
  assert(num_inputs == 1);
  int out_channels;
  ActiMode activation;
  RegularizerMode kernel_reg_type;
  float kernel_reg_lambda;
  bool use_bias;
  DataType data_type;
  DataType quantization_type;
  bool offload;
  size_t id, transformer_layer_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  LayerID layer_guid(id, transformer_layer_id);
  dez.deserialize(out_channels);
  dez.deserialize(activation);
  dez.deserialize(kernel_reg_type);
  dez.deserialize(kernel_reg_lambda);
  dez.deserialize(use_bias);
  dez.deserialize(data_type);
  dez.deserialize(quantization_type);
  dez.deserialize(offload);

  LinearParams params;
  params.activation = activation;
  params.kernel_reg_type = kernel_reg_type;
  params.kernel_reg_lambda = kernel_reg_lambda;
  params.out_channels = out_channels;
  params.use_bias = use_bias;
  params.data_type = data_type;
  params.layer_guid = layer_guid;
  params.quantization_type = quantization_type;
  params.offload = offload;
  return ff.get_or_create_node<Linear>(inputs[0], params);
}

LinearParams Linear::get_params() const {
  LinearParams params;
  params.layer_guid = this->layer_guid;
  params.out_channels = this->out_channels;
  params.use_bias = this->use_bias;
  params.data_type = this->data_type;
  params.activation = this->activation;
  params.kernel_reg_type = this->kernel_reg_type;
  params.kernel_reg_lambda = this->kernel_reg_lambda;
  params.quantization_type = this->quantization_type;
  params.offload = this->offload;

  return params;
}

bool LinearParams::is_valid(ParallelTensorShape const &input_shape) const {
  ParallelTensorShape output_shape, kernel_shape, bias_shape;
  this->solve_dims(input_shape,
                   output_shape.dims,
                   &output_shape.num_dims,
                   kernel_shape.dims,
                   &kernel_shape.num_dims,
                   bias_shape.dims,
                   &bias_shape.num_dims);
  bool is_valid = true;
  is_valid &= input_shape.is_valid();
  is_valid &= output_shape.is_valid();
  is_valid &= kernel_shape.is_valid();
  if (use_bias) {
    is_valid &= bias_shape.is_valid();
  }
  return is_valid;
}

void LinearParams::solve_dims(const ParallelTensor input,
                              ParallelDim output_dims[MAX_TENSOR_DIM],
                              int *output_ndims,
                              ParallelDim kernel_dims[MAX_TENSOR_DIM],
                              int *kernel_ndims,
                              ParallelDim bias_dims[MAX_TENSOR_DIM],
                              int *bias_ndims) const {
  this->solve_dims(input->get_shape(),
                   output_dims,
                   output_ndims,
                   kernel_dims,
                   kernel_ndims,
                   bias_dims,
                   bias_ndims);
}

void LinearParams::solve_dims(ParallelTensorShape const &input_shape,
                              ParallelTensorShape &output_shape,
                              ParallelTensorShape &kernel_shape,
                              ParallelTensorShape &bias_shape) const {
  this->solve_dims(input_shape,
                   output_shape.dims,
                   &output_shape.num_dims,
                   kernel_shape.dims,
                   &kernel_shape.num_dims,
                   bias_shape.dims,
                   &bias_shape.num_dims);
}

void LinearParams::solve_dims(ParallelTensorShape const &input_shape,
                              ParallelDim output_dims[MAX_TENSOR_DIM],
                              int *output_ndims,
                              ParallelDim kernel_dims[MAX_TENSOR_DIM],
                              int *kernel_ndims,
                              ParallelDim bias_dims[MAX_TENSOR_DIM],
                              int *bias_ndims) const {
  assert((output_dims == nullptr) == (output_ndims == nullptr));
  assert((kernel_dims == nullptr) == (kernel_ndims == nullptr));
  assert((bias_dims == nullptr) == (bias_ndims == nullptr));

  std::vector<ParallelDimMappingRecord> mapping;
  this->construct_mappings(mapping, input_shape);
  this->mark_replica_dims(input_shape, output_dims, kernel_dims, bias_dims);

  solve_parallel_dim_mappings(
      mapping, {input_shape.dims}, {kernel_dims, bias_dims}, {output_dims});

  this->calculate_nonreplica_dim_sizes(input_shape,
                                       output_dims,
                                       output_ndims,
                                       kernel_dims,
                                       kernel_ndims,
                                       bias_dims,
                                       bias_ndims);
}

std::unordered_map<LinearParams::NamedDimensions, int>
    LinearParams::get_dimension_names(
        ParallelTensorShape const &input_shape) const {
  int num_dims = input_shape.num_dims;

  return {{INPUT_CHANNEL, 0},
          {INPUT_SAMPLE, num_dims - 2},
          {INPUT_REPLICA, num_dims - 1},
          {OUTPUT_CHANNEL, 0},
          {OUTPUT_SAMPLE, num_dims - 2},
          {OUTPUT_REPLICA, num_dims - 1},
          {KERNEL_CHANNEL_IN, 0},
          {KERNEL_CHANNEL_OUT, 1},
          {BIAS_CHANNEL_OUT, 0}};
}

void LinearParams::calculate_nonreplica_dim_sizes(
    ParallelTensorShape const &input_shape,
    ParallelDim output_dims[MAX_TENSOR_DIM],
    int *output_ndims,
    ParallelDim kernel_dims[MAX_TENSOR_DIM],
    int *kernel_ndims,
    ParallelDim bias_dims[MAX_TENSOR_DIM],
    int *bias_ndims) const {
  auto dimension_names = this->get_dimension_names(input_shape);
  int num_dims = input_shape.num_dims;

  if (output_dims != nullptr) {
    for (int i = 1; i < input_shape.num_dims - 1; i++) {
      output_dims[i].size = input_shape.dims[i].size;
    }
    output_dims[dimension_names.at(OUTPUT_CHANNEL)].size = this->out_channels;
    *output_ndims = num_dims;
  }
  if (kernel_dims != nullptr) {
    kernel_dims[dimension_names.at(KERNEL_CHANNEL_IN)].size =
        input_shape.dims[INPUT_CHANNEL].size /
        input_shape.dims[INPUT_CHANNEL].degree;
    kernel_dims[dimension_names.at(KERNEL_CHANNEL_OUT)].size =
        this->out_channels;
    *kernel_ndims = num_dims;
  }
  if (bias_dims != nullptr) {
    bias_dims[dimension_names.at(BIAS_CHANNEL_OUT)].size = this->out_channels;
    *bias_ndims = num_dims;
  }
}

void LinearParams::mark_replica_dims(
    ParallelTensorShape const &input_shape,
    ParallelDim output_dims[MAX_TENSOR_DIM],
    ParallelDim kernel_dims[MAX_TENSOR_DIM],
    ParallelDim bias_dims[MAX_TENSOR_DIM]) const {
  int num_dims = input_shape.num_dims;
  auto dimension_names = this->get_dimension_names(input_shape);
  if (output_dims != nullptr) {
    output_dims[dimension_names.at(OUTPUT_REPLICA)].is_replica_dim = true;
  }
  if (kernel_dims != nullptr) {
    for (int i = 2; i < num_dims; i++) {
      kernel_dims[i].is_replica_dim = true;
    }
  }
  if (bias_dims != nullptr) {
    for (int i = 1; i < num_dims; i++) {
      bias_dims[i].is_replica_dim = true;
    }
  }
}

void LinearParams::construct_mappings(
    std::vector<ParallelDimMappingRecord> &mappings,
    ParallelTensorShape const &input_shape) const {
  std::unordered_map<NamedDimensions, int> dimension_names =
      this->get_dimension_names(input_shape);

  Op::construct_output_parallel_dims(
      mappings,
      {{dimension_names.at(INPUT_CHANNEL), dimension_names.at(OUTPUT_REPLICA)},
       {dimension_names.at(INPUT_REPLICA),
        dimension_names.at(OUTPUT_CHANNEL)}});
  for (int i = 1; i < input_shape.num_dims - 1; i++) {
    Op::construct_output_parallel_dims(mappings, i, i);
  }

  Op::construct_weight_parallel_dims(mappings,
                                     {{dimension_names.at(INPUT_CHANNEL),
                                       dimension_names.at(KERNEL_CHANNEL_IN)},
                                      {dimension_names.at(INPUT_REPLICA),
                                       dimension_names.at(KERNEL_CHANNEL_OUT)}},
                                     0 /*input_idx*/,
                                     KERNEL_IDX);
  // map a bunch of replica dimensions for the unnamed dimensions in the input
  for (int i = 1; i < input_shape.num_dims - 1; i++) {
    Op::construct_weight_parallel_dims(
        mappings, i, i + 1, 0 /*input_idx*/, KERNEL_IDX);
  }

  Op::construct_weight_parallel_dims(mappings,
                                     {
                                         {dimension_names.at(INPUT_REPLICA),
                                          dimension_names.at(BIAS_CHANNEL_OUT)},
                                     },
                                     0 /*input_idx*/,
                                     BIAS_IDX);
  for (int i = 0; i < input_shape.num_dims - 1; i++) {
    Op::construct_weight_parallel_dims(
        mappings, i, i + 1, 0 /*input_idx*/, BIAS_IDX);
  }
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::LinearParams>::operator()(
    FlexFlow::LinearParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.out_channels);
  hash_combine(key, params.use_bias);
  hash_combine(key, params.data_type);
  hash_combine(key, params.activation);
  hash_combine(key, params.kernel_reg_type);
  hash_combine(key, params.kernel_reg_lambda);
  hash_combine(key, params.quantization_type);
  hash_combine(key, params.offload);
  return key;
}
}; // namespace std
