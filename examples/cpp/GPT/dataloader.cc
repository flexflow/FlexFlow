#include "minigpt.h"

using namespace Legion;

LegionRuntime::Logger::Category log_app("Transformer");

DataLoader::DataLoader(FFModel &ff,
                       MiniGPTConfig const &minigptconfig,
                       Tensor const & input,
                       Tensor const & label) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  num_samples = 0;
  log_app.print("Use random dataset...");
  num_samples =
      ff.config.batchSize * ff.config.workersPerNode * ff.config.numNodes;
  log_app.print("Number of random samples = %d\n", num_samples);
  return;
  {
    batch_input = input;
    int num_dims = input->num_dims

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
    full_label = ff.create_parallel_tensor_legion_ordering(num_dims, dims, DT_FLOAT);
    ff.map_tensor(full_label, NULL /*parallel_op*/);
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
  runtime->execute_task(ctx, launcher);
}

void DataLoader::load_entire_dataset(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  assert(regions.size() == 2);
  assert(task->regions.size() == 2);
  // get input and label pointer
  float *input_ptr = helperGetTensorPointerWO<float>(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  int *label_ptr = helperGetTensorPointerWO<int>(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  read_mnist(input_ptr, label_ptr)
}

void DataLoader::next_batch(FFModel &ff) {
  return;
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
  // progress next_index
  next_index += ff.config.batchSize;
}

void DataLoader::reset() {
  next_index = 0;
}


void read_mnist(float *input_ptr, int *label_ptr) {
  // read inputs
  std::ifstream input("train-images-idx3-ubyte", std::ios::binary);
  if (input.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    int n_rows = 0;
    int n_cols = 0;
    input.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    input.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);
    input.read((char *)&n_rows, sizeof(n_rows));
    n_rows = reverseInt(n_rows);
    input.read((char *)&n_cols, sizeof(n_cols));
    n_cols = reverseInt(n_cols);

    for (int i = 0; i < number_of_images; i++) {
      for (int r = 0; r < n_rows; r++) {
        for (int c = 0; c < n_cols; c++) {
          unsigned char temp = 0;
          input.read((char *)&temp, sizeof(temp));
          input_ptr[i * n_rows * n_cols + r * n_cols + c] =
              (float)temp / 255.0f;
        }
      }
    }
  } else {
    std::cout << "Error opening MNIST input data file" << std::endl;
    assert(false);
  }

  // read labels
  std::ifstream labels("train-labels-idx1-ubyte", std::ios::binary);
  if (labels.is_open()) {
    int magic_number = 0;
    int number_of_images = 0;
    labels.read((char *)&magic_number, sizeof(magic_number));
    magic_number = reverseInt(magic_number);
    labels.read((char *)&number_of_images, sizeof(number_of_images));
    number_of_images = reverseInt(number_of_images);

    for (int i = 0; i < number_of_images; i++) {
      unsigned char temp = 0;
      labels.read((char *)&temp, sizeof(temp));
      label_ptr[i] = temp;
    }
  } else {
    std::cout << "Error opening MNIST label data file" << std::endl;
    assert(false);
  }
}




void FlexFlow::register_custom_tasks() {
  // Load entire dataset
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
  // Load label
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_2,
                                   "Load Labels");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_label>(
        registrar, "Load Label Task");
  }
  // Load weights
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_3,
                                   "Load Weights");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_weights>(
        registrar, "Load Label Task");
  }

  // Load bias
  {
    TaskVariantRegistrar registrar(CUSTOM_GPU_TASK_ID_4,
                                   "Load bias");
    registrar.add_constraint(ProcessorConstraint(Processor::TOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<DataLoader::load_bias>(
        registrar, "Load Label Task");
  }
}