#include "test_utils.h"

#define  PRECISION 6
#define MAX_DATASET_PATH_LEN 1023

struct ArgsConfig {
  char dataset_path[MAX_DATASET_PATH_LEN];
  char data_type[30];
  int num_dim;
};

void initialize_tensor_from_file(const std::string file_path, 
    Tensor label, 
    const FFModel& ff, 
    std::string data_type, 
    int num_dim);

void initialize_tensor_gradient_from_file(const std::string file_path, 
    Tensor label, 
    const FFModel& ff, 
    std::string data_type,  int num_dim) {
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  ArgsConfig args_config;
  strcpy(args_config.dataset_path, file_path.c_str());
  strcpy(args_config.data_type, data_type.c_str());
  if (num_dim == 1) {
    TaskLauncher launcher(
        INIT_TENSOR_1D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    // regions[0]: full_sparse_input
    launcher.add_region_requirement(
        RegionRequirement(label.region_grad,
                          WRITE_ONLY, EXCLUSIVE, label.region_grad,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (num_dim == 2) {
    TaskLauncher launcher(
        INIT_TENSOR_2D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    // regions[0]: full_sparse_input
    launcher.add_region_requirement(
        RegionRequirement(label.region_grad,
                          WRITE_ONLY, EXCLUSIVE, label.region_grad,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (num_dim == 3) {
    TaskLauncher launcher(
        INIT_TENSOR_3D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    // regions[0]: full_sparse_input
    launcher.add_region_requirement(
        RegionRequirement(label.region_grad,
                          WRITE_ONLY, EXCLUSIVE, label.region_grad,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (num_dim == 4) {
    TaskLauncher launcher(
        INIT_TENSOR_4D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    // regions[0]: full_sparse_input
    launcher.add_region_requirement(
        RegionRequirement(label.region_grad,
                          WRITE_ONLY, EXCLUSIVE, label.region_grad,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else {
    throw 255;
  }

}


void initialize_tensor_from_file(const std::string file_path, 
    Tensor label, 
    const FFModel& ff, 
    std::string data_type,  int num_dim) {
  Context ctx = ff.config.lg_ctx;
  Runtime* runtime = ff.config.lg_hlr;
  ArgsConfig args_config;
  strcpy(args_config.dataset_path, file_path.c_str());
  strcpy(args_config.data_type, data_type.c_str());
  if (num_dim == 1) {
    TaskLauncher launcher(
        INIT_TENSOR_1D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    launcher.add_region_requirement(
        RegionRequirement(label.region,
                          WRITE_ONLY, EXCLUSIVE, label.region,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (num_dim == 2) {
    TaskLauncher launcher(
        INIT_TENSOR_2D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    launcher.add_region_requirement(
        RegionRequirement(label.region,
                          WRITE_ONLY, EXCLUSIVE, label.region,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (num_dim == 3) {
    TaskLauncher launcher(
        INIT_TENSOR_3D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    launcher.add_region_requirement(
        RegionRequirement(label.region,
                          WRITE_ONLY, EXCLUSIVE, label.region,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (num_dim == 4) {
    TaskLauncher launcher(
        INIT_TENSOR_4D_FROM_FILE_CPU_TASK,
        TaskArgument(&args_config, sizeof(args_config)));
    launcher.add_region_requirement(
        RegionRequirement(label.region,
                          WRITE_ONLY, EXCLUSIVE, label.region,
                          MAP_TO_FB_MEMORY));
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else {
    throw 255;
  }

}


template<int DIM>
void initialize_tensor_from_file_task(const Task *task,
    const std::vector<PhysicalRegion> &regions,
    Context ctx,
    Runtime* runtime) {
  const ArgsConfig args_config = *((const ArgsConfig *)task->args);
  std::string file_path((const char*)args_config.dataset_path);
  std::string data_type((const char*)args_config.data_type);
  Rect<DIM> rect_label_tensor = runtime->get_index_space_domain(
      ctx, task->regions[0].region.get_index_space());
  if (data_type == "int") {
    const AccessorWO<int, DIM> acc_label_tensor(regions[0], FID_DATA);
    int* tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
    std::fstream myfile(file_path, std::ios_base::in);
    int a;
    int i = 0;
    while (myfile >> a)
    {
      tensor_ptr[i] = a;
      i++;
    }   
    myfile.close();
  } else if (data_type == "float") {
    const AccessorWO<float, DIM> acc_label_tensor(regions[0], FID_DATA);
    float* tensor_ptr = acc_label_tensor.ptr(rect_label_tensor.lo);
    std::fstream myfile(file_path, std::ios_base::in);
    float a;
    int i = 0;
    while (myfile >> a)
    {
      tensor_ptr[i] = a;
      i++;
    } 
    myfile.close();
  }
}


void dump_region_to_file(FFModel &ff, 
    LogicalRegion &region, 
    std::string file_path, 
    int dims) {
  Context ctx = ff.config.lg_ctx;
  Runtime *runtime = ff.config.lg_hlr;
  ArgsConfig args_config;
  strcpy(args_config.dataset_path, file_path.c_str());
  if (dims == 2) {
    TaskLauncher launcher(DUMP_TENSOR_2D_CPU_TASK, 
                          TaskArgument(&args_config, sizeof(args_config)));
    launcher.add_region_requirement(
      RegionRequirement(
        region, READ_WRITE, EXCLUSIVE, region, MAP_TO_ZC_MEMORY)
    );
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);
  } else if (dims == 3) {
    TaskLauncher launcher(DUMP_TENSOR_3D_CPU_TASK, 
                          TaskArgument(&args_config, sizeof(args_config)));
    launcher.add_region_requirement(
      RegionRequirement(
        region, READ_WRITE, EXCLUSIVE, region, MAP_TO_ZC_MEMORY)
    );
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);

  } else if (dims == 4) {
    TaskLauncher launcher(DUMP_TENSOR_4D_CPU_TASK, 
                          TaskArgument(&args_config, sizeof(args_config)));
    launcher.add_region_requirement(
      RegionRequirement(
        region, READ_WRITE, EXCLUSIVE, region, MAP_TO_ZC_MEMORY)
    );
    launcher.add_field(0, FID_DATA);
    runtime->execute_task(ctx, launcher);

  } else
  {
    std::cout << "dims: " << dims << std::endl;
    // not supported
    throw 255;
  }
}


template<int DIM>
void dump_tensor_task(const Task* task,
    const std::vector<PhysicalRegion>& regions,
    Context ctx, Runtime* runtime) {
  assert(task->regions.size() == 1);
  assert(regions.size() == 1);
  const ArgsConfig args_config = *((const ArgsConfig *)task->args);
  std::string file_path((const char*)args_config.dataset_path);
  const AccessorRO<float, DIM> acc_tensor(regions[0], FID_DATA);
  Rect<DIM> rect_fb = runtime->get_index_space_domain(
    ctx, task->regions[0].region.get_index_space());
  assert(acc_tensor.accessor.is_dense_arbitrary(rect_fb));
  const float* tensor_ptr = acc_tensor.ptr(rect_fb.lo);
  std::ofstream myfile;
  myfile.open (file_path);
  for (size_t i = 0; i < rect_fb.volume(); ++i) {
    // printf("%.6lf ", (float)tensor_ptr[i]);
    myfile << std::fixed << std::setprecision(PRECISION) << (float)tensor_ptr[i] << " ";
  }
  myfile.close();
}





template void dump_tensor_task<1>(const Task* task,
                      const std::vector<PhysicalRegion>& regions,
                      Context ctx, Runtime* runtime);
template void dump_tensor_task<2>(const Task* task,
                      const std::vector<PhysicalRegion>& regions,
                      Context ctx, Runtime* runtime);
template void dump_tensor_task<3>(const Task* task,
                      const std::vector<PhysicalRegion>& regions,
                      Context ctx, Runtime* runtime);
template void dump_tensor_task<4>(const Task* task,
                      const std::vector<PhysicalRegion>& regions,
                      Context ctx, Runtime* runtime);
template void initialize_tensor_from_file_task<1>(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime* runtime);
template void initialize_tensor_from_file_task<2>(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime* runtime);
template void initialize_tensor_from_file_task<3>(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime* runtime);
template void initialize_tensor_from_file_task<4>(const Task *task,
                    const std::vector<PhysicalRegion> &regions,
                    Context ctx,
                    Runtime* runtime);

