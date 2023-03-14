#include "minigpt.h"
#include <random>


using namespace Legion;


DataLoader::DataLoader(FFModel &ff,
                       MiniGPTConfig const *minigptconfig,
                       ParallelTensor const & input,
                       ParallelTensor const & pos,
                       ParallelTensor const & label) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  num_samples = 0;
  num_samples =
      ff.config.batchSize * ff.config.workersPerNode * ff.config.numNodes;
  
  {
    batch_input = input;
    int num_dims = input->num_dims;
    

    ParallelDim dims[num_dims];
    for (int i = 0; i < num_dims; i++) {
      dims[i].size = input->dims[i].size;
      dims[i].degree = 1;
      dims[i].parallel_idx = -1;
      dims[i].is_replica_dim = input->dims[i].is_replica_dim;
      // Assume only the first dim can be the replica dim
      assert(i == num_dims - 1 || (!dims[i].is_replica_dim));
    }
    dims[num_dims - 2].size = num_samples;
    full_input = ff.create_parallel_tensor_legion_ordering(num_dims, dims, DT_FLOAT);
    ff.map_tensor(full_input, NULL /*parallel_op*/);
  }
  {
    batch_label = label;
    int num_dims = label->num_dims;
    ParallelDim dims[num_dims];
    for (int i = 0; i < num_dims; i++) {
      dims[i].size = label->dims[i].size;
      dims[i].degree = 1;
      dims[i].parallel_idx = -1;
      dims[i].is_replica_dim = label->dims[i].is_replica_dim;
      // Assume only the first dim can be the replica dim
      assert(i == num_dims - 1 || (!dims[i].is_replica_dim));
    }
    dims[num_dims - 2].size = num_samples;
    full_label = ff.create_parallel_tensor_legion_ordering(num_dims, dims, DT_INT32);
    ff.map_tensor(full_label, NULL /*parallel_op*/);
  }
  {
    batch_pos = pos;
    int num_dims = pos->num_dims;
    ParallelDim dims[num_dims];
    for (int i = 0; i < num_dims; i++) {
      dims[i].size = pos->dims[i].size;
      dims[i].degree = 1;
      dims[i].parallel_idx = -1;
      dims[i].is_replica_dim = pos->dims[i].is_replica_dim;
      // Assume only the first dim can be the replica dim
      assert(i == num_dims - 1 || (!dims[i].is_replica_dim));
    }
    dims[num_dims - 2].size = num_samples;
    full_pos = ff.create_parallel_tensor_legion_ordering(num_dims, dims, DT_FLOAT);
    ff.map_tensor(full_pos, NULL /*parallel_op*/);
  }
  // Load entire dataset
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1, TaskArgument(NULL, 0));
   // regions[1]: full_input
  launcher.add_region_requirement(
      RegionRequirement(full_input->region,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        full_input->region,
                        MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  // regions[1]: full_label
  launcher.add_region_requirement(
      RegionRequirement(full_label->region,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        full_label->region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(1, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(full_pos->region,
                        WRITE_ONLY,
                        EXCLUSIVE,
                        full_pos->region,
                        MAP_TO_ZC_MEMORY));
  launcher.add_field(2, FID_DATA);
  runtime->execute_task(ctx, launcher);
}

void DataLoader::load_entire_dataset(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  assert(regions.size() == 3);
  assert(task->regions.size() == 3);

  AccessorWO<float, 4> const acc_input(regions[0], FID_DATA);
  AccessorWO<int, 4> const acc_label(regions[1], FID_DATA);
  AccessorWO<float, 4> const acc_pos(regions[2], FID_DATA);
  Rect<4> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  Rect<4> rect_label = runtime->get_index_space_domain(
      ctx, task->regions[1].region.get_index_space());
  Rect<4> rect_pos = runtime->get_index_space_domain(
      ctx, task->regions[2].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));
  assert(acc_label.accessor.is_dense_arbitrary(rect_label));
  assert(acc_pos.accessor.is_dense_arbitrary(rect_pos));
  float *input_ptr = acc_input.ptr(rect_input.lo);
  int *label_ptr = acc_label.ptr(rect_label.lo);
  float *pos_ptr = acc_pos.ptr(rect_pos.lo);

  std::random_device device; // obtain a random number from hardware
  std::mt19937 gen(device()); // seed the generator
  std::uniform_int_distribution<> distr(0, 3); // define the range


  for (size_t i = 0; i < rect_input.volume(); i++) {
    input_ptr[i] = ((float)std::rand()) / RAND_MAX;
  }
  for (size_t i = 0; i < rect_label.volume(); i++) {
    label_ptr[i] = distr(gen);
  }
  for (size_t i = 0; i < rect_pos.volume(); i++) {
    pos_ptr[i] = ((float)std::rand()) / RAND_MAX;
  }
 
  // // get input and label pointer
  // int *input_ptr = helperGetTensorPointerWO<int>(
  //     regions[0], task->regions[0], FID_DATA, ctx, runtime);
  // int *label_ptr = helperGetTensorPointerWO<int>(
  //     regions[1], task->regions[1], FID_DATA, ctx, runtime);
  // int *pos_ptr = helperGetTensorPointerWO<int>(
  //     regions[2], task->regions[2], FID_DATA, ctx, runtime);
  
  // for(int i = 0; i < 16; i++){
  // std::random_device device; // obtain a random number from hardware
  // std::mt19937 gen(device()); // seed the generator
  // std::uniform_int_distribution<> distr(0, 30000); // define the range
  // input_ptr[i] = distr(gen);
  // pos_ptr[i] = i;
  // label_ptr[i] = distr(gen);
}



void DataLoader::next_batch(FFModel &ff) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  // Load Input
  {
    Domain domain = runtime->get_index_space_domain(
        ctx, batch_input->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % batch_input->dims[2].size ==
             0);
      meta.num_samples =
          ff.config.batchSize / batch_input->dims[2].size;
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = idx++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_1,
                           batch_input->parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           batch_input->machine_view.hash());
    launcher.add_region_requirement(
        RegionRequirement(full_input->region,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          full_input->region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_input->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_input->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load Labels
  {
    Domain domain = runtime->get_index_space_domain(
        ctx, batch_label->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % batch_label->dims[2].size ==
             0);
      meta.num_samples =
          ff.config.batchSize / batch_label->dims[2].size;
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = idx++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_2,
                           batch_label->parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           batch_label->machine_view.hash());
    // Full dataset in ZCM
    launcher.add_region_requirement(
        RegionRequirement(full_label->region,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          full_label->region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_label->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_label->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // Load pos
  {
    Domain domain = runtime->get_index_space_domain(
        ctx, batch_pos->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % batch_pos->dims[2].size ==
             0);
      meta.num_samples =
          ff.config.batchSize / batch_pos->dims[2].size;
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = idx++;
      }
      argmap.set_point(*it, TaskArgument(&meta, sizeof(SampleIdxs)));
    }
    IndexLauncher launcher(CUSTOM_GPU_TASK_ID_3,
                           batch_pos->parallel_is,
                           TaskArgument(NULL, 0),
                           argmap,
                           Predicate::TRUE_PRED,
                           false /*must*/,
                           0 /*mapper_id*/,
                           batch_pos->machine_view.hash());
    // Full dataset in ZCM
    launcher.add_region_requirement(
        RegionRequirement(full_pos->region,
                          0 /*projection id*/,
                          READ_ONLY,
                          EXCLUSIVE,
                          full_pos->region,
                          MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(
        RegionRequirement(batch_pos->part,
                          0 /*projection id*/,
                          WRITE_ONLY,
                          EXCLUSIVE,
                          batch_pos->region));
    launcher.add_field(1, FID_DATA);
    runtime->execute_index_space(ctx, launcher);
  }
  // progress next_index
  next_index += ff.config.batchSize;
}

void DataLoader::reset() {
  next_index = 0;
}








void FlexFlow::register_custom_tasks() {
  //Load entire dataset
  {
    TaskVariantRegistrar registrar(CUSTOM_CPU_TASK_ID_1, "Load Entire Dataset");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_entire_dataset>(
        registrar, "Load Entire Dataset Task");
  }
  // Load input
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_1, "Load Inputs");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_input>(
        registrar, "Load Inputs Task");
  }
  //Load label
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_2,
                                   "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_label>(
        registrar, "Load Label Task");
  }
  // Load pos
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_3,
                                   "Load pos");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_pos>(
        registrar, "Load Label Task");
  }

  // // Load bias
  // {
  //   TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_4,
  //                                  "Load bias");
  //   registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
  //   registrar.set_leaf();
  //   Runtime::preregister_task_variant<DataLoader::load_bias>(
  //       registrar, "Load Label Task");
  // }
}