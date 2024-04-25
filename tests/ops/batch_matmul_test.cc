#include "model.h"
#include "test_utils.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace Legion;
Legion::Logger log_app("bmm_test");

struct BMMTestMeta {
  int m, k, n, d;
  BMMTestMeta(int _m, int _k, int _n, int _d) {
    m = _m, k = _k, n = _n, d = _d;
  }
};

BMMTestMeta get_test_meta(const std::string file_path) {
  std::fstream myfile(file_path, std::ios_base::in);
  int m, k, n, d;
  myfile >> m >> k >> n >> d;
  return BMMTestMeta(m, k, n, d);
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
  Tensor dense_input1;
  {
    int const dims[3] = {
        test_meta.d, test_meta.k, test_meta.m}; // target shape (d,k,m)
    // HACK: have to pass "batch_matmul" 3-dimensional strategy string id to
    // tell FF to distribute this tensor correctly
    dense_input1 = ff.create_tensor<3>(dims, "batch_matmul", DT_FLOAT);
  }
  Tensor dense_input2;
  {
    int const dims[3] = {
        test_meta.d, test_meta.k, test_meta.n}; // shape (n,k,d)
    // HACK: have to pass "batch_matmul" 3-dimensional strategy string id to
    // tell FF to distribute this tensor correctly
    dense_input2 = ff.create_tensor<3>(dims, "batch_matmul", DT_FLOAT);
  }
  // build batch matmul layer
  Tensor batch_matmul_ret = ff.batch_matmul("batch_matmul",
                                            dense_input1,
                                            dense_input2,
                                            true /* trans_a */,
                                            false /* trans_b */);
  // load inputs tensors and output gradients tensors for testing
  auto input1_file_path = "test_input1.txt";
  auto input2_file_path = "test_input2.txt";
  auto output_grad_file_path = "test_output_grad.txt";
  initialize_tensor_from_file(input1_file_path, dense_input1, ff, "float", 3);
  initialize_tensor_from_file(input2_file_path, dense_input2, ff, "float", 3);
  initialize_tensor_gradient_from_file(
      output_grad_file_path, batch_matmul_ret, ff, "float", 3);
  // run forward and backward to produce results
  ff.init_layers();
  ff.forward();
  ff.backward();
  // dump results to file for python validation
  dump_region_to_file(ff, batch_matmul_ret.region, "output.txt", 3);
  dump_region_to_file(ff, dense_input1.region_grad, "input1_grad.txt", 3);
  dump_region_to_file(ff, dense_input2.region_grad, "input2_grad.txt", 3);
}

void register_custom_tasks() {
  // std::cout <<
  // static_cast<std::underlying_type<TaskIDs>::type>(ZERO_INIT_TASK_ID) <<
  // std::endl;
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
