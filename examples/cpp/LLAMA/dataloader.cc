#include "llama.h"
#include <random>

using namespace Legion;

DataLoader::DataLoader(FFModel &ff,
                       LLAMAConfig const *llamaconfig,
                       ParallelTensor const &input) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  num_samples = 347;

  {
    batch_input = input;
    int num_dims = input->num_dims;
    std::cout << "before input shape"
              << "\n";
    for (int i = 0; i < input->num_dims; i++) {
      std::cout << input->dims[i].size << "------\n";
    }

    ParallelDim dims[num_dims];
    for (int i = 0; i < num_dims; i++) {
      dims[i].size = input->dims[i].size;
      dims[i].degree = 1;
      dims[i].parallel_idx = -1;
      dims[i].is_replica_dim = input->dims[i].is_replica_dim;
      // Assume only the first dim can be the replica dim
      assert(i == num_dims - 1 || (!dims[i].is_replica_dim));
    }
    dims[num_dims - 1].size = num_samples;
    full_input =
        ff.create_parallel_tensor_legion_ordering(num_dims, dims, DT_INT64);

    std::cout << "input shape"
              << "\n";
    for (int i = 0; i < full_input->num_dims; i++) {
      std::cout << full_input->dims[i].size << "------\n";
    }
    ff.map_tensor(full_input, NULL /*parallel_op*/);
  }

  // Load entire dataset
  TaskLauncher launcher(CUSTOM_CPU_TASK_ID_1, TaskArgument(NULL, 0));
  // regions[1]: full_input
  launcher.add_region_requirement(RegionRequirement(full_input->region,
                                                    WRITE_ONLY,
                                                    EXCLUSIVE,
                                                    full_input->region,
                                                    MAP_TO_FB_MEMORY));
  launcher.add_field(0, FID_DATA);
  runtime->execute_task(ctx, launcher);
}





void DataLoader::load_entire_dataset(Task const *task,
                                     std::vector<PhysicalRegion> const &regions,
                                     Context ctx,
                                     Runtime *runtime) {
  assert(regions.size() == 1);
  assert(task->regions.size() == 1);

  AccessorWO<long, 3> const acc_input(regions[0], FID_DATA);
  Rect<3> rect_input = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  assert(acc_input.accessor.is_dense_arbitrary(rect_input));

  long *input_ptr = acc_input.ptr(rect_input.lo);
  std::cout << "load entire dataset" << rect_input.volume();

  // load from file
  load_from_file(input_ptr,
                 rect_input.volume(),
                 "/home/ubuntu/FlexFlow/examples/cpp/LLAMA/tokens/input");
}

void DataLoader::next_batch(FFModel &ff) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  fprintf(stderr, "----------next batch--------------");
  // Load Input
  {
    Domain domain =
        runtime->get_index_space_domain(ctx, batch_input->parallel_is);
    ArgumentMap argmap;
    int idx = next_index;
    for (Domain::DomainPointIterator it(domain); it; it++) {
      SampleIdxs meta;
      assert(ff.config.batchSize % batch_input->dims[1].size == 0);
      meta.num_samples = ff.config.batchSize / batch_input->dims[2].size;
      for (int i = 0; i < meta.num_samples; i++) {
        meta.idxs[i] = idx++;
        meta.token_idx = next_token_idx;
        meta.batch_idx = next_batch_index;
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
    launcher.add_region_requirement(RegionRequirement(full_input->region,
                                                      0 /*projection id*/,
                                                      READ_ONLY,
                                                      EXCLUSIVE,
                                                      full_input->region,
                                                      MAP_TO_ZC_MEMORY));
    launcher.add_field(0, FID_DATA);
    launcher.add_region_requirement(RegionRequirement(batch_input->part,
                                                      0 /*projection id*/,
                                                      READ_WRITE,
                                                      EXCLUSIVE,
                                                      batch_input->region));
    launcher.add_field(1, FID_DATA);

    fprintf(stderr, "----------lunach the input--------------");
    runtime->execute_index_space(ctx, launcher);
  }
  // progress next_index
  next_index += ff.config.batchSize;
  next_token_idx += 1;
}

void DataLoader::reset() {
  next_index = 0;
  next_token_idx = 0;
  next_batch_index = 0;
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
}