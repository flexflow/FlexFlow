#include "flexflow/ops/reduce.h"
#include "flexflow/model.h"
#include "flexflow/utils/hash_utils.h"
#include "legion/legion_utilities.h"

namespace FlexFlow {
// declare Legion names
using Legion::ArgumentMap;
using Legion::Context;
using Legion::coord_t;
using Legion::Domain;
using Legion::FutureMap;
using Legion::IndexLauncher;
using Legion::PhysicalRegion;
using Legion::Predicate;
using Legion::Rect;
using Legion::RegionRequirement;
using Legion::Runtime;
using Legion::Task;
using Legion::TaskArgument;
using Legion::TaskLauncher;

bool operator==(ReduceParams const &lhs, ReduceParams const &rhs) {
  return (lhs.axes == rhs.axes) && (lhs.keepdims == rhs.keepdims);
}

bool ReduceParams::is_valid(ParallelTensorShape const &input) const {
  for (size_t i = 0; i < axes.size(); i++) {
    if (axes[i] >= input.num_dims) {
      return false;
    }
  }
  return input.is_valid();
}

ReduceParams Reduce::get_params() const {
  ReduceParams params;
  params.axes.clear();
  for (int i = 0; i < num_axes; i++) {
    params.axes.push_back(this->axes[i]);
  }
  params.keepdims = keepdims;
  params.layer_guid = this->layer_guid;
  if (this->name != nullptr) {
    strcpy(params.name, this->name);
  }
  return params;
}

Tensor FFModel::reduce_sum(const Tensor input,
                           std::vector<int> const &_axes,
                           bool keepdims,
                           char const *name) {
  Layer *rd = new Layer(this,
                        OP_REDUCE_SUM,
                        DT_FLOAT,
                        name,
                        1 /*input*/,
                        0 /*weights*/,
                        1 /*outputs*/,
                        input);
  // Use Legion indexing to store axes
  std::vector<int> axes;
  for (size_t i = 0; i < _axes.size(); i++) {
    axes.push_back(input->num_dims - 1 - _axes[i]);
  }
  int dims[MAX_TENSOR_DIM];
  int numdim = input->num_dims;
  if (keepdims) {
    for (int i = 0; i < input->num_dims; i++) {
      dims[i] = input->dims[i];
    }
    for (size_t i = 0; i < axes.size(); i++) {
      dims[axes[i]] = 1;
    }
  } else {
    numdim = 0;
    for (int i = 0; i < input->num_dims; i++) {
      bool reduced = false;
      for (size_t j = 0; j < axes.size(); j++) {
        if (axes[j] == i) {
          reduced = true;
        }
      }
      if (!reduced) {
        dims[numdim++] = input->dims[i];
      }
    }
    assert(numdim + axes.size() == input->num_dims);
  }
  rd->outputs[0] = create_tensor_legion_ordering(
      numdim, dims, input->data_type, rd, 0, true /*create_grad*/);
  rd->add_int_vector_property("legion_axes", axes);
  rd->add_int_property("keepdims", keepdims);
  layers.push_back(rd);
  return rd->outputs[0];
}

Op *Reduce::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  std::vector<int> axes;
  long long value;
  layer->get_int_vector_property("legion_axes", axes);
  layer->get_int_property("keepdims", value);
  bool keepdims = value;
  return new Reduce(
      model, layer->layer_guid, inputs[0], axes, keepdims, layer->name);
}

Reduce::Reduce(FFModel &model,
               ReduceParams const &params,
               const ParallelTensor input,
               char const *name)
    : Reduce(model,
             params.layer_guid,
             input,
             params.axes,
             params.keepdims,
             params.name) {}

Reduce::Reduce(FFModel &model,
               LayerID const &_layer_guid,
               const ParallelTensor input,
               std::vector<int> const &_axes,
               bool _keepdims,
               char const *name)
    : Op(model,
         OP_REDUCE_SUM,
         input->data_type,
         name,
         1 /*inputs*/,
         0 /*weights*/,
         1 /*outputs*/,
         input),
      num_axes(_axes.size()), keepdims(_keepdims) {
  layer_guid = _layer_guid;
  for (size_t i = 0; i < num_axes; i++) {
    axes[i] = _axes[i];
  }
  int num_dims = input->num_dims;
  ParallelDim dims[MAX_TENSOR_DIM];
  if (keepdims) {
    num_dims = input->num_dims;
    for (int i = 0; i < num_dims; i++) {
      dims[i] = input->dims[i];
    }
    for (int i = 0; i < num_axes; i++) {
      // Currently assume that we cannot parallelize along reduced dims
      assert(dims[axes[i]].degree == 1);
      dims[axes[i]].size = 1;
    }
  } else {
    num_dims = 0;
    for (int i = 0; i < input->num_dims; i++) {
      bool reduced = false;
      for (int j = 0; j < num_axes; j++) {
        if (axes[j] == i) {
          reduced = true;
        }
      }
      if (!reduced) {
        dims[num_dims++] = input->dims[i];
      } else {
        // Currently assume that we cannot parallelize along reduced dims
        assert(input->dims[i].degree == 1);
        assert(input->dims[i].parallel_idx == -1);
      }
    }
  }
  outputs[0] = model.create_parallel_tensor_legion_ordering(
      num_dims, dims, input->data_type, this);
}

void Reduce::init(FFModel const &ff) {
  assert(check_output_input_weight_same_parallel_is());
  parallel_is = outputs[0]->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_init(ff, argmap);
  IndexLauncher launcher(REDUCE_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(Reduce)),
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
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap(ff, fm);
};

OpMeta *Reduce::init_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  Reduce *rd = (Reduce *)task->args;
  FFHandler handle = *((FFHandler *)task->local_args);
  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      DT_FLOAT, regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      DT_FLOAT, regions[1], task->regions[1], FID_DATA, ctx, runtime);
  ReduceMeta *m = new ReduceMeta(handle, rd, input.domain);
  std::strcpy(m->op_name, rd->name);
  m->layer_guid = rd->layer_guid;
  return m;
}

void Reduce::forward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_forward(ff, argmap);
  IndexLauncher launcher(REDUCE_FWD_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, false),
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
  runtime->execute_index_space(ctx, launcher);
}

void Reduce::forward_task(Task const *task,
                          std::vector<PhysicalRegion> const &regions,
                          Context ctx,
                          Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  ReduceMeta const *m = *((ReduceMeta **)task->local_args);
  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      DT_FLOAT, regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorWO(
      DT_FLOAT, regions[1], task->regions[1], FID_DATA, ctx, runtime);

  Reduce::forward_kernel_wrapper(m, input, output);
}

void Reduce::backward(FFModel const &ff) {
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  set_argumentmap_for_backward(ff, argmap);
  IndexLauncher launcher(REDUCE_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         outputs[0]->machine_view.hash());
  // regions[0](I): output_grad
  launcher.add_region_requirement(RegionRequirement(outputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    outputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  // regions[1](I/O): input_grad
  launcher.add_region_requirement(RegionRequirement(inputs[0]->part_grad,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    inputs[0]->region_grad));
  launcher.add_field(1, FID_DATA);
  runtime->execute_index_space(ctx, launcher);
}

void Reduce::backward_task(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  ReduceMeta const *m = *((ReduceMeta **)task->local_args);
  GenericTensorAccessorR output_grad = helperGetGenericTensorAccessorRO(
      DT_FLOAT, regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
      DT_FLOAT, regions[1], task->regions[1], FID_DATA, ctx, runtime);
  Reduce::backward_kernel_wrapper(m, output_grad, input_grad);
}

bool Reduce::measure_operator_cost(Simulator *sim,
                                   MachineView const &mv,
                                   CostMetrics &cost_metrics) const {
  ParallelTensorBase sub_input, sub_output;
  if (!outputs[0]->get_sub_tensor(mv, sub_output)) {
    return false;
  }
  if (!inputs[0]->get_sub_tensor(mv, sub_input)) {
    return false;
  }
  ReduceMeta *m = new ReduceMeta(sim->handler, this, sub_input.get_domain());
  sim->free_all();
  float *input_ptr = (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
  assert(input_ptr != NULL);
  cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
  GenericTensorAccessorR input_acc(
      inputs[0]->data_type, sub_input.get_domain(), input_ptr);

  float *output_ptr = (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
  assert(output_ptr != NULL);
  cost_metrics.outputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
  GenericTensorAccessorW output_acc(
      outputs[0]->data_type, sub_output.get_domain(), output_ptr);

  assert(m->profiling == false);

  std::function<void()> forward, backward;
  forward = [&] { forward_kernel_wrapper(m, input_acc, output_acc); };
  if (sim->computationMode == COMP_MODE_TRAINING) {
    float *input_grad_ptr =
        (float *)sim->allocate(sub_input.get_volume(), DT_FLOAT);
    assert(input_grad_ptr != NULL);
    cost_metrics.inputs_memory += cost_metrics.total_mem_diff_from(sim->offset);
    GenericTensorAccessorW input_grad_acc(
        inputs[0]->data_type, sub_input.get_domain(), input_grad_ptr);

    float *output_grad_ptr =
        (float *)sim->allocate(sub_output.get_volume(), DT_FLOAT);
    assert(output_grad_ptr != NULL);
    cost_metrics.outputs_memory +=
        cost_metrics.total_mem_diff_from(sim->offset);
    GenericTensorAccessorR output_grad_acc(
        outputs[0]->data_type, sub_output.get_domain(), output_grad_ptr);

    backward = [=] {
      backward_kernel_wrapper(m, output_grad_acc, input_grad_acc);
    };
  }

  inner_measure_operator_cost(sim, forward, backward, cost_metrics);

  if (sim->computationMode == COMP_MODE_TRAINING) {
    printf("[Measure Reduce] name(%s) forward_time(%.4lf) "
           "backward_time(%.4lf)\n",
           name,
           cost_metrics.forward_time,
           cost_metrics.backward_time);
  } else {
    printf("[Measure Reduce] name(%s) forward_time(%.4lf)\n",
           name,
           cost_metrics.forward_time);
  }

  return true;
}

void Reduce::serialize(Legion::Serializer &sez) const {
  ReduceParams params = get_params();
  sez.serialize(params.axes.size());
  for (size_t i = 0; i < params.axes.size(); i++) {
    sez.serialize(params.axes[i]);
  }
  sez.serialize(params.keepdims);
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

using PCG::Node;
Node Reduce::deserialize(FFModel &ff,
                         Legion::Deserializer &dez,
                         ParallelTensor inputs[],
                         int num_inputs) {
  assert(num_inputs == 1);
  size_t axes_size;
  bool keepdims;
  std::vector<int> axes;
  dez.deserialize(axes_size);
  for (size_t i = 0; i < axes_size; i++) {
    int dim_idx;
    dez.deserialize(dim_idx);
    axes.push_back(dim_idx);
  }
  dez.deserialize(keepdims);
  size_t id, transformer_layer_id, deserialized_model_id;
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);

  return ff.get_or_create_node<Reduce>(inputs[0], {axes, keepdims, layer_guid});
}

Op *Reduce::materialize(FFModel &ff,
                        ParallelTensor inputs[],
                        int num_inputs) const {
  ReduceParams params = get_params();
  return new Reduce(ff, params, inputs[0], this->name);
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ReduceParams>::operator()(
    FlexFlow::ReduceParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.axes.size());
  for (int n : params.axes) {
    hash_combine(key, n);
  }
  hash_combine(key, params.keepdims);
  hash_combine(key, params.layer_guid.id);
  return key;
}
}; // namespace std
