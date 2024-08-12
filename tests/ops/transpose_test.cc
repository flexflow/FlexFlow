#include "model.h"
#include "test_utils.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace Legion;
Realm::Logger log_app("transpose_test");

struct TransposeTestMeta {
  int m, k, d;
  TransposeTestMeta(int _m, int _k, int _d) {
    m = _m, k = _k, d = _d;
  }
};

TransposeTestMeta get_test_meta(const std::string file_path) {
  std::fstream myfile(file_path, std::ios_base::in);
  int m, k, d;
  myfile >> m >> k >> d;
  return TransposeTestMeta(m, k, d);
}

void top_level_task(Task const *task,
                    std::vector<PhysicalRegion> const &regions,
                    Context ctx,
                    Runtime *runtime) {
  // std::cout<< "test framework launched" << std::endl;
  auto test_meta = get_test_meta("test_meta.txt");
  FFConfig ffConfig;
  // create ff model object
  FFModel ff(ffConfig);
  // create input tensor
  Tensor dense_input;
  {
    int const dims[3] = {
        test_meta.d, test_meta.m, test_meta.k}; // target shape (d,m,k)
    dense_input = ff.create_tensor<3>(dims, "transpose", DT_FLOAT);
  }
  // build transpose layer
  Tensor ret = ff.transpose("transpose", dense_input);
  // load inputs tensors and output gradients tensors for testing
  auto input1_file_path = "test_input1.txt";
  auto output_grad_file_path = "test_output_grad.txt";
  initialize_tensor_from_file(input1_file_path, dense_input, ff, "float", 3);
  initialize_tensor_gradient_from_file(
      output_grad_file_path, ret, ff, "float", 3);
  // run forward and backward to produce results
  ff.init_layers();
  ff.forward();
  ff.backward();
  // dump results to file for python validation
  dump_region_to_file(ff, ret.region, "output.txt", 3);
  dump_region_to_file(ff, dense_input.region_grad, "input1_grad.txt", 3);
}

void register_custom_tasks() {
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_2D_FROM_FILE_CPU_TASK,
                                   "Load 2d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<2>>(
        registrar, "Load 2d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_3D_FROM_FILE_CPU_TASK,
                                   "Load 3d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<3>>(
        registrar, "Load 3d tensor Task");
  }
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_4D_FROM_FILE_CPU_TASK,
                                   "Load 4d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<4>>(
        registrar, "Load 4d tensor Task");
  }

  {
    TaskVariantRegistrar registrar(DUMP_TENSOR_2D_CPU_TASK, "Compare Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<2>>(
        registrar, "Compare Tensor Task");
  }
  {
    TaskVariantRegistrar registrar(DUMP_TENSOR_4D_CPU_TASK, "Compare Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<4>>(
        registrar, "Compare Tensor Task");
  }
  {
    TaskVariantRegistrar registrar(DUMP_TENSOR_3D_CPU_TASK, "Compare Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<3>>(
        registrar, "Compare Tensor Task");
  }
}
