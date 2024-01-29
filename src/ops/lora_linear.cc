#include "flexflow/ops/lora_linear.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/layer.h"
#include "flexflow/model.h"
#include "flexflow/ops/kernels/lora_linear_kernels.h"
#include "flexflow/utils/hash_utils.h"
#include "flexflow/utils/peft_weight_allocator.h"
#include "legion/legion_utilities.h"
#include <sys/stat.h>
#include <sys/types.h>
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "flexflow/utils/cuda_helper.h"
#else
#include "flexflow/utils/hip_helper.h"
#endif

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

using namespace FlexFlow::Kernels::LoraLinear;

void FFModel::lora_linear(Tensor const input,
                          Tensor const output,
                          OperatorType op_type,
                          char const *name) {
  assert(input->data_type == output->data_type);
  Layer *lora = nullptr;
  lora = new Layer(this,
                   op_type,
                   output->data_type,
                   name,
                   2 /*inputs*/,
                   0 /*weights*/,
                   1 /*outputs*/,
                   input,
                   output);
  {
    int numdims = output->num_dims;
    int dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdims; i++) {
      dims[i] = output->dims[i];
    }
    lora->outputs[0] = create_tensor_legion_ordering(
        numdims, dims, output->data_type, lora, 0, true /*create_grad*/);
  }
  layers.push_back(lora);
}

Op *LoraLinear::create_operator_from_layer(
    FFModel &model,
    Layer const *layer,
    std::vector<ParallelTensor> const &inputs) {
  return new LoraLinear(model,
                        layer->layer_guid,
                        layer->op_type,
                        inputs[0],
                        inputs[1],
                        layer->name);
}

LoraLinear::LoraLinear(FFModel &model,
                       LoraLinear const &other,
                       ParallelTensor const input,
                       ParallelTensor const output)
    : LoraLinear(
          model, other.layer_guid, other.op_type, input, output, other.name) {}

LoraLinear::LoraLinear(FFModel &model,
                       Params const &params,
                       Input const &inputs,
                       char const *name)
    : LoraLinear(model,
                 params.layer_guid,
                 params.type,
                 inputs.first,
                 inputs.second,
                 params.name) {}

LoraLinear::LoraLinear(FFModel &model,
                       LayerID const &_layer_guid,
                       OperatorType _op_type,
                       ParallelTensor const _input,
                       ParallelTensor const _output,
                       char const *name)
    : Op(model,
         _op_type,
         _output->data_type,
         name,
         2 /*inputs*/,
         0 /*weights*/,
         false,
         1 /*outputs*/,
         _input,
         _output) {
  assert(_input->data_type == _output->data_type);
  // overwrite layer_guid
  layer_guid = _layer_guid;
  data_type = _output->data_type;

  ParallelTensorShape input_shape = this->inputs[0]->get_shape();
  LoraLinearParams params = this->get_params();

  // Create output tensor
  {
    int numdim = inputs[1]->num_dims;
    ParallelDim dims[MAX_TENSOR_DIM];
    for (int i = 0; i < numdim; i++) {
      dims[i] = inputs[1]->dims[i];
    }
    outputs[0] = model.create_parallel_tensor_legion_ordering(
        numdim, dims, inputs[1]->data_type, this);
  }
  // assert(check_output_input_weight_parallel_dims(allocate_weights));
}

void LoraLinear::init(FFModel const &ff) {
  assert(false && "LoraLinear does not support normal init");
}

void LoraLinear::init_inference(
    FFModel const &ff,
    std::vector<ParallelTensor> const &batch_inputs,
    std::vector<ParallelTensor> const &batch_outputs,
    MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  assert(batch_inputs.size() == 2);
  assert(batch_outputs.size() == 1);
  // Assert that the output and the second input are mapped to the same
  // region/part
  assert(batch_outputs[0]->region == batch_inputs[1]->region);
  assert(batch_outputs[0]->part == batch_inputs[1]->part);
  // assert(check_output_input_weight_same_machine_view());
  // output is considered as an input to allow in-place optimization
  ParallelTensor output_tensor = batch_outputs[0];
  parallel_is = output_tensor->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = mv ? mv : &output_tensor->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_init_inference(ff, argmap, output_tensor);
  IndexLauncher launcher(LORA_LINEAR_INIT_TASK_ID,
                         parallel_is,
                         TaskArgument(this, sizeof(LoraLinear)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[0]->part,
                                                    0 /*projection id*/,
                                                    READ_ONLY,
                                                    EXCLUSIVE,
                                                    batch_inputs[0]->region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
  set_opmeta_from_futuremap_inference(ff, fm, output_tensor);
}

/*
  regions[0](O): output
  regions[1](I): kernel
  regions[2](I): bias
*/
OpMeta *LoraLinear::init_task(Task const *task,
                              std::vector<PhysicalRegion> const &regions,
                              Context ctx,
                              Runtime *runtime) {
  LoraLinear const *lora = (LoraLinear *)task->args;
  FFHandler handle = *((FFHandler const *)task->local_args);
  GenericTensorAccessorR input =
      helperGetGenericTensorAccessorRO(lora->inputs[0]->data_type,
                                       regions[0],
                                       task->regions[0],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  GenericTensorAccessorW output =
      helperGetGenericTensorAccessorRW(lora->outputs[0]->data_type,
                                       regions[1],
                                       task->regions[1],
                                       FID_DATA,
                                       ctx,
                                       runtime);
  int in_dim = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  int out_dim = output.domain.hi()[0] - output.domain.lo()[0] + 1;
  int batch_size = output.domain.get_volume() / out_dim;
  assert(input.domain.get_volume() == in_dim * batch_size);
  assert(output.domain.get_volume() == out_dim * batch_size);

  LoraLinearMeta *m = new LoraLinearMeta(handle, lora);
  m->trainable_inputs[0] = lora->trainable_inputs[0];
  std::strcpy(m->op_name, lora->name);
  m->layer_guid = lora->layer_guid;

  return m;
}

struct LoraLinearRegisterInfo {
  LoraLinear const *lora;
  PEFTModelID model_id;
  LoraLinearConfig lora_config;
};

void LoraLinear::register_peft_model(
    FFModel const &ff,
    std::vector<ParallelTensor> const &batch_inputs,
    std::vector<ParallelTensor> const &batch_outputs,
    PEFTModelID const &model_id,
    LoraLinearConfig const lora_config) {
  assert(check_output_input_weight_same_parallel_is());
  assert(batch_inputs.size() == 2);
  assert(batch_outputs.size() == 1);
  // Assert that the output and the second input are mapped to the same
  // region/part
  assert(batch_outputs[0]->region == batch_inputs[1]->region);
  assert(batch_outputs[0]->part == batch_inputs[1]->part);
  // assert(check_output_input_weight_same_machine_view());
  // output is considered as an input to allow in-place optimization
  ParallelTensor output_tensor = batch_outputs[0];
  parallel_is = output_tensor->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = &output_tensor->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_inference(ff, argmap, output_tensor);
  LoraLinearRegisterInfo info;
  info.lora = this;
  info.model_id = model_id;
  info.lora_config = lora_config;
  IndexLauncher launcher(LORA_LINEAR_REG_TASK_ID,
                         parallel_is,
                         TaskArgument(&info, sizeof(LoraLinearRegisterInfo)),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  FutureMap fm = runtime->execute_index_space(ctx, launcher);
  fm.wait_all_results();
}

template <typename DT>
void load_peft_from_file(
    DT *ptr, size_t size, bool sharded, int shard_id, std::string filepath) {
  std::ifstream in(filepath, std::ios::in | std::ios::binary);
  if (!in.good()) {
    printf("Could not open file: %s\n", filepath.c_str());
  }
  assert(in.good() && "incorrect weight file path");
  std::vector<DT> host_array(size);
  size_t target_data_size = sizeof(DT) * size;
  in.seekg(sharded * shard_id * target_data_size, in.beg);
  in.read((char *)host_array.data(), target_data_size);

  size_t in_get_size = in.gcount();
  if (in_get_size != target_data_size) {
    printf("load weight data error: %lu, %lu, %lu\n",
           in_get_size,
           target_data_size,
           sizeof(DT));
    assert(false);
  }
  assert(size == host_array.size());
  copy_tensor_host_to_dev(ptr, host_array.data(), size);
  in.close();
}

void LoraLinear::register_model_task(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  LoraLinearRegisterInfo const *info =
      static_cast<LoraLinearRegisterInfo const *>(task->args);
  LoraLinearMeta *m = *((LoraLinearMeta **)task->local_args);
  LoraLinear const *lora = info->lora;

  int shard_id = task->index_point.point_data[0];

  int rank = info->lora_config.rank;
  int num_dims = lora->inputs[0]->num_dims;
  int in_dim = lora->inputs[0]->dims[0].size / lora->inputs[0]->dims[0].degree;
  int out_dim = lora->inputs[1]->dims[0].size / lora->inputs[1]->dims[0].degree;
  int w0_num_elements = rank * in_dim;
  int w1_num_elements = rank * out_dim;

  DataType dt = m->input_type[0];
  assert(dt == m->input_type[1]);
  assert(dt == m->output_type[0]);
  assert(dt == lora->inputs[0]->data_type);
  assert(dt == lora->inputs[1]->data_type);
  assert(dt == lora->outputs[0]->data_type);
  assert(m->model_weights.find(info->model_id) == m->model_weights.end());

  LoraLinearWeight weight;
  weight.in_dim = in_dim;
  weight.out_dim = out_dim;
  weight.rank = rank;
  PEFTWeightAllocator *allocator = m->handle.peft_weight_allocator;
  weight.w0_ptr = allocator->allocate_local_weights_untyped(
      info->model_id, w0_num_elements * data_type_size(dt));
  weight.w1_ptr = allocator->allocate_local_weights_untyped(
      info->model_id, w1_num_elements * data_type_size(dt));

  // get layer name
  assert(lora->name != nullptr &&
         "Layer name is not set, cannot determine weights location");
  std::string lora_layername = std::string(lora->name);
  std::string searchString = "lora";
  size_t found = lora_layername.find(searchString);
  if (found == std::string::npos) {
    std::cout << "LoraLinear layer name not in the right format (does not "
                 "contain word 'lora')"
              << std::endl;
    assert(false);
  }
  std::string lora_layername_substr =
      lora_layername.substr(0, found + searchString.length());

  // load weights from file
  std::string weights_folder_filepath = join_path({
      info->lora_config.cache_folder,
      "weights",
      info->lora_config.peft_model_id,
      dt == DT_FLOAT ? "full-precision" : "half-precision",
  });
  std::string w0_filepath =
      join_path({weights_folder_filepath, lora_layername_substr + "_A_weight"});
  std::string w1_filepath =
      join_path({weights_folder_filepath, lora_layername_substr + "_B_weight"});
  if (dt == DT_FLOAT) {
    std::cout << "Loading LORA weight " << lora_layername_substr + "_A_weight"
              << ", size: " << w0_num_elements << ", shard: " << shard_id
              << std::endl;
    load_peft_from_file(
        (float *)weight.w0_ptr, w0_num_elements, true, shard_id, w0_filepath);
    std::cout << "Loading LORA weight " << lora_layername_substr + "_B_weight"
              << ", size: " << w1_num_elements << ", shard: " << shard_id
              << std::endl;
    load_peft_from_file(
        (float *)weight.w1_ptr, w1_num_elements, false, shard_id, w1_filepath);
  } else if (dt == DT_HALF) {
    std::cout << "Loading LORA weight " << lora_layername_substr + "_A_weight"
              << ", size: " << w0_num_elements << ", shard: " << shard_id
              << std::endl;
    load_peft_from_file(
        (half *)weight.w0_ptr, w0_num_elements, true, shard_id, w0_filepath);
    std::cout << "Loading LORA weight " << lora_layername_substr + "_B_weight"
              << ", size: " << w1_num_elements << ", shard: " << shard_id
              << std::endl;
    load_peft_from_file(
        (half *)weight.w1_ptr, w1_num_elements, false, shard_id, w1_filepath);
  } else {
    assert(false && "Data type not supported");
  }

  if (lora->inputs[0]->dims[num_dims - 1].degree == 1) {
    // Input is partitioned (no replication)
    // w0_grad is local weight gradients
    weight.w0_grad_ptr = allocator->allocate_local_weights_untyped(
        info->model_id, w0_num_elements * data_type_size(dt));
    // w1_grad is sync weight gradients
    weight.w1_grad_ptr = allocator->allocate_sync_weights_untyped(
        info->model_id, w1_num_elements * data_type_size(dt));
  } else {
    // Input is replicated
    // w0_grad is sync weight gradients
    weight.w0_grad_ptr = allocator->allocate_sync_weights_untyped(
        info->model_id, w0_num_elements * data_type_size(dt));
    // w1_grad is local weight gradients
    weight.w1_grad_ptr = allocator->allocate_local_weights_untyped(
        info->model_id, w1_num_elements * data_type_size(dt));
  }
  m->model_weights[info->model_id] = weight;
}

void LoraLinear::forward(FFModel const &ff) {
  assert(false && "LoraLinear does not support normal init");
}

FutureMap
    LoraLinear::inference(FFModel const &ff,
                          BatchConfigFuture const &bc,
                          std::vector<ParallelTensor> const &batch_inputs,
                          std::vector<ParallelTensor> const &batch_outputs,
                          MachineView const *mv) {
  assert(check_output_input_weight_same_parallel_is());
  assert(batch_inputs.size() == 2);
  assert(batch_outputs.size() == 1);
  // Assert that the output and the second input are mapped to the same
  // region/part
  assert(batch_outputs[0]->region == batch_inputs[1]->region);
  assert(batch_outputs[0]->part == batch_inputs[1]->part);
  // assert(check_output_input_weight_same_machine_view());
  // output is considered as an input to allow in-place optimization
  ParallelTensor output_tensor = batch_outputs[0];
  parallel_is = output_tensor->parallel_is;
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  MachineView const *view = mv ? mv : &output_tensor->machine_view;
  size_t machine_view_hash = view->hash();
  set_argumentmap_for_inference(ff, argmap, output_tensor);
  IndexLauncher launcher(LORA_LINEAR_INF_TASK_ID,
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
  launcher.add_region_requirement(RegionRequirement(batch_inputs[1]->part,
                                                    0 /*projection id*/,
                                                    READ_WRITE,
                                                    EXCLUSIVE,
                                                    batch_inputs[1]->region));
  launcher.add_field(1, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

void LoraLinear::inference_task(Task const *task,
                                std::vector<PhysicalRegion> const &regions,
                                Context ctx,
                                Runtime *runtime) {
  LoraLinearMeta *m = *((LoraLinearMeta **)task->local_args);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_active_tokens() == 0) {
    return;
  }
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());
  assert(m->input_type[0] == m->output_type[0]);

  GenericTensorAccessorR input = helperGetGenericTensorAccessorRO(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorW output = helperGetGenericTensorAccessorRW(
      m->input_type[1], regions[1], task->regions[1], FID_DATA, ctx, runtime);
  // int in_dim = input.domain.hi()[0] - input.domain.lo()[0] + 1;
  // int out_dim = output.domain.hi()[0] - output.domain.lo()[0] + 1;

  // int num_infr_tokens = bc->num_active_infr_tokens();
  // int num_peft_tokens = bc->num_active_peft_tokens();
  inference_kernel_wrapper(m, bc, input, output);

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];

    // Check if output directory exists, and create it if it does not
    char const *folder_path = "./inference_tensors/";
    struct stat st = {0};
    if (stat(folder_path, &st) == -1) {
      // Directory does not exist, create it
      mkdir(folder_path, 0700);
    }

    std::string lora_layername = std::string(m->op_name);
    std::string searchString = "lora";
    size_t found = lora_layername.find(searchString);
    if (found == std::string::npos) {
      std::cout << "LoraLinear layer name not in the right format (does not "
                   "contain word 'lora')"
                << std::endl;
      assert(false);
    }
    std::string lora_layername_substr =
        lora_layername.substr(0, found + searchString.length());

    // output base filepath, shared by all tensors from the same operator
    std::string base_filepath = std::string(folder_path);
    if (m->layer_guid.model_id > 0) {
      base_filepath += "model_" + std::to_string(m->layer_guid.model_id) + "_";
    }
    base_filepath += "fwd_step_" + std::to_string(m->decoding_step);
    base_filepath +=
        "_layers_" + std::to_string(m->layer_guid.transformer_layer_id) + "_" +
        lora_layername_substr + "_shard_" + std::to_string(shard_id);

    // save batch config, if passed
    if (bc != nullptr) {
      bc->save_to_file(base_filepath + "_batch_config");
    }

    std::string filename = base_filepath + "_input_" + std::to_string(0);
    if (input.data_type == DT_FLOAT) {
      save_tensor(
          input.get_float_ptr(), input.domain.get_volume(), filename.c_str());
    } else if (input.data_type == DT_HALF) {
      save_tensor(
          input.get_half_ptr(), input.domain.get_volume(), filename.c_str());
    } else {
      assert(false);
    }

    // std::cout << "base_filepath: " << base_filepath << std::endl;
    // std::cout << "m->decoding_step: " << m->decoding_step << std::endl;
    if (m->decoding_step == 0) {
      for (auto it = m->model_weights.begin(); it != m->model_weights.end();
           ++it) {
        PEFTModelID peft_model_id = it->first;
        LoraLinearWeight weight = m->model_weights[peft_model_id];
        std::string filenameA = base_filepath + "_weight_A";
        std::string filenameB = base_filepath + "_weight_B";
        if (m->input_type[0] == DT_FLOAT) {
          save_tensor((float *)weight.w0_ptr,
                      weight.rank * weight.in_dim,
                      filenameA.c_str());
          save_tensor((float *)weight.w1_ptr,
                      weight.rank * weight.out_dim,
                      filenameB.c_str());
        } else if (m->input_type[0] == DT_HALF) {
          save_tensor((half *)weight.w0_ptr,
                      weight.rank * weight.in_dim,
                      filenameA.c_str());
          save_tensor((half *)weight.w1_ptr,
                      weight.rank * weight.out_dim,
                      filenameB.c_str());
        } else {
          assert(false && "Data type not supported");
        }
      }
    }

    filename = base_filepath + "_output_" + std::to_string(0);
    if (output.data_type == DT_FLOAT) {
      save_tensor(
          output.get_float_ptr(), output.domain.get_volume(), filename.c_str());
    } else if (output.data_type == DT_HALF) {
      save_tensor(
          output.get_half_ptr(), output.domain.get_volume(), filename.c_str());
    } else {
      assert(false);
    }
    m->decoding_step++;
  }
}

FutureMap LoraLinear::peft_bwd(FFModel const &ff,
                               BatchConfigFuture const &bc,
                               std::vector<ParallelTensor> const &batch_inputs,
                               std::vector<ParallelTensor> const &batch_outputs,
                               MachineView const *mv) {
  assert(batch_inputs.size() == 2);
  assert(batch_outputs.size() == 1);
  // Assert that the output and the second input are mapped to the same
  // region/part
  assert(batch_outputs[0]->region == batch_inputs[1]->region);
  assert(batch_outputs[0]->part == batch_inputs[1]->part);
  ArgumentMap argmap;
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  ParallelTensor output_tensor = batch_outputs[0];
  parallel_is = output_tensor->parallel_is;
  MachineView const *view = mv ? mv : &output_tensor->machine_view;
  set_argumentmap_for_inference(ff, argmap, output_tensor);
  size_t machine_view_hash = view->hash();
  IndexLauncher launcher(LORA_LINEAR_PEFT_BWD_TASK_ID,
                         parallel_is,
                         TaskArgument(nullptr, 0),
                         argmap,
                         Predicate::TRUE_PRED,
                         false /*must*/,
                         0 /*mapper_id*/,
                         machine_view_hash);
  launcher.add_future(bc);
  launcher.add_region_requirement(
      RegionRequirement(batch_inputs[0]->part_grad,
                        0 /*projection id*/,
                        reset_input_grads[0] ? WRITE_ONLY : READ_WRITE,
                        EXCLUSIVE,
                        batch_inputs[0]->region_grad));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(batch_inputs[1]->part_grad,
                        0 /*projection id*/,
                        reset_input_grads[1] ? WRITE_ONLY : READ_WRITE,
                        EXCLUSIVE,
                        batch_inputs[1]->region_grad));
  launcher.add_field(1, FID_DATA);
  return runtime->execute_index_space(ctx, launcher);
}

void LoraLinear::peft_bwd_task(Task const *task,
                               std::vector<PhysicalRegion> const &regions,
                               Context ctx,
                               Runtime *runtime) {
  Domain input_domain = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  LoraLinearMeta *m = *((LoraLinearMeta **)task->local_args);
  BatchConfig const *bc = BatchConfig::from_future(task->futures[0]);
  if (bc->num_active_peft_tokens() == 0) {
    return;
  }
  assert(regions.size() == 2);
  assert(task->regions.size() == regions.size());
  assert(m->input_type[0] == m->output_type[0]);

  GenericTensorAccessorW input_grad = helperGetGenericTensorAccessorRW(
      m->input_type[0], regions[0], task->regions[0], FID_DATA, ctx, runtime);
  GenericTensorAccessorR output_grad = helperGetGenericTensorAccessorRO(
      m->output_type[0], regions[1], task->regions[1], FID_DATA, ctx, runtime);

  // int in_dim = input_grad.domain.hi()[0] - input_grad.domain.lo()[0] + 1;
  // int out_dim = output_grad.domain.hi()[0] - output_grad.domain.lo()[0] + 1;
  // int num_infr_tokens = bc->num_active_infr_tokens();
  // int num_peft_tokens = bc->num_active_peft_tokens();
  peft_bwd_kernel_wrapper(m, bc, input_grad, output_grad);

  if (m->inference_debugging) {
    assert(task->index_point.get_dim() == 1);
    int shard_id = task->index_point.point_data[0];

    // Check if output directory exists, and create it if it does not
    char const *folder_path = "./inference_tensors/";
    struct stat st = {0};
    if (stat(folder_path, &st) == -1) {
      // Directory does not exist, create it
      mkdir(folder_path, 0700);
    }

    std::string lora_layername = std::string(m->op_name);
    std::string searchString = "lora";
    size_t found = lora_layername.find(searchString);
    if (found == std::string::npos) {
      std::cout << "LoraLinear layer name not in the right format (does not "
                   "contain word 'lora')"
                << std::endl;
      assert(false);
    }
    std::string lora_layername_substr =
        lora_layername.substr(0, found + searchString.length());

    // output base filepath, shared by all tensors from the same operator
    std::string base_filepath = std::string(folder_path);
    if (m->layer_guid.model_id > 0) {
      base_filepath += "model_" + std::to_string(m->layer_guid.model_id) + "_";
    }
    base_filepath += "bwd_step_" + std::to_string(m->bwd_step);
    base_filepath +=
        "_layers_" + std::to_string(m->layer_guid.transformer_layer_id) + "_" +
        lora_layername_substr + "_shard_" + std::to_string(shard_id);

    // save batch config, if passed
    if (bc != nullptr) {
      bc->save_to_file(base_filepath + "_batch_config");
    }

    std::string filename = base_filepath + "_input_" + std::to_string(0);
    if (input_grad.data_type == DT_FLOAT) {
      save_tensor(input_grad.get_float_ptr(),
                  input_grad.domain.get_volume(),
                  filename.c_str());
    } else if (input_grad.data_type == DT_HALF) {
      save_tensor(input_grad.get_half_ptr(),
                  input_grad.domain.get_volume(),
                  filename.c_str());
    } else {
      assert(false);
    }

    // std::cout << "base_filepath: " << base_filepath << std::endl;
    // std::cout << "m->decoding_step: " << m->decoding_step << std::endl;
    if (m->bwd_step == 0) {
      for (auto it = m->model_weights.begin(); it != m->model_weights.end();
           ++it) {
        PEFTModelID peft_model_id = it->first;
        LoraLinearWeight weight = m->model_weights[peft_model_id];
        std::string filenameA = base_filepath + "_weight_A";
        std::string filenameB = base_filepath + "_weight_B";
        if (m->input_type[0] == DT_FLOAT) {
          save_tensor((float *)weight.w0_grad_ptr,
                      weight.rank * weight.in_dim,
                      filenameA.c_str());
          save_tensor((float *)weight.w1_grad_ptr,
                      weight.rank * weight.out_dim,
                      filenameB.c_str());
        } else if (m->input_type[0] == DT_HALF) {
          save_tensor((half *)weight.w0_grad_ptr,
                      weight.rank * weight.in_dim,
                      filenameA.c_str());
          save_tensor((half *)weight.w1_grad_ptr,
                      weight.rank * weight.out_dim,
                      filenameB.c_str());
        } else {
          assert(false && "Data type not supported");
        }
      }
    }

    filename = base_filepath + "_output_" + std::to_string(0);
    if (output_grad.data_type == DT_FLOAT) {
      save_tensor(output_grad.get_float_ptr(),
                  output_grad.domain.get_volume(),
                  filename.c_str());
    } else if (output_grad.data_type == DT_HALF) {
      save_tensor(output_grad.get_half_ptr(),
                  output_grad.domain.get_volume(),
                  filename.c_str());
    } else {
      assert(false);
    }
    m->bwd_step++;
  }
}

void LoraLinear::backward(FFModel const &ff) {
  assert(false && "LoraLinear does not support normal backward");
}

void LoraLinear::print_layer(FFModel const &ff) {}

void LoraLinear::map_output_tensors(FFModel &ff) {
  assert(numOutputs == 1);
  assert(numInputs == 2);
  assert(outputs[0]->get_volume() == inputs[1]->get_volume());
  outputs[0]->parallel_is = inputs[1]->parallel_is;
  outputs[0]->region = inputs[1]->region;
  outputs[0]->part = inputs[1]->part;
  outputs[0]->region_grad = inputs[1]->region_grad;
  outputs[0]->part_grad = inputs[1]->part_grad;
}

bool LoraLinear::measure_operator_cost(Simulator *sim,
                                       MachineView const &mv,
                                       CostMetrics &cost_metrics) const {
  return false;
}

bool operator==(LoraLinearParams const &lhs, LoraLinearParams const &rhs) {
  return lhs.layer_guid == rhs.layer_guid && lhs.type == rhs.type;
}

void LoraLinear::serialize(Legion::Serializer &sez) const {
  sez.serialize(this->layer_guid.id);
  sez.serialize(this->layer_guid.transformer_layer_id);
  sez.serialize(this->layer_guid.model_id);
  sez.serialize(this->op_type);
  sez.serialize(strlen(this->name));
  sez.serialize(this->name, strlen(this->name));
}

/* static */
using PCG::Node;
Node LoraLinear::deserialize(FFModel &ff,
                             Legion::Deserializer &dez,
                             ParallelTensor inputs[],
                             int num_inputs) {
  assert(num_inputs == 2);
  size_t id, transformer_layer_id, deserialized_model_id;
  OperatorType op_type;
  size_t name_len;
  char name[MAX_OPNAME] = {0};
  dez.deserialize(id);
  dez.deserialize(transformer_layer_id);
  dez.deserialize(deserialized_model_id);
  dez.deserialize(op_type);
  dez.deserialize(name_len);
  dez.deserialize(name, name_len);
  LayerID layer_guid(id, transformer_layer_id, deserialized_model_id);

  LoraLinearParams params;
  params.layer_guid = layer_guid;
  params.type = op_type;
  strcpy(params.name, name);
  return ff.get_or_create_node<LoraLinear>({inputs[0], inputs[1]}, params);
}

Op *LoraLinear::materialize(FFModel &ff,
                            ParallelTensor inputs[],
                            int num_inputs) const {
  LoraLinearParams params = get_params();
  return new LoraLinear(ff, params, {inputs[0], inputs[1]}, this->name);
}

LoraLinearParams LoraLinear::get_params() const {
  LoraLinearParams params;
  params.layer_guid = this->layer_guid;
  params.type = this->op_type;
  if (this->name != nullptr) {
    strcpy(params.name, this->name);
  }
  return params;
}

bool LoraLinearParams::is_valid(
    std::pair<ParallelTensorShape, ParallelTensorShape> const &input_shape)
    const {
  return true;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::LoraLinearParams>::operator()(
    FlexFlow::LoraLinearParams const &params) const {
  size_t key = 0;
  hash_combine(key, params.layer_guid.id);
  hash_combine(key, params.layer_guid.transformer_layer_id);
  hash_combine(key, params.layer_guid.model_id);
  return key;
}
}; // namespace std
