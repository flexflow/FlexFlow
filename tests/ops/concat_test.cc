#include "model.h"
#include "test_utils.h"
#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>
using namespace Legion;
Legion::Logger log_app("concat_test");

struct ConcatTestMeta {
  int batch_size, i_dim, num_channels, projected_num_channels,
      dense_projection_i_dim;
  ConcatTestMeta(int _batch_size,
                 int _i_dim,
                 int _num_channels,
                 int _projected_num_channels,
                 int _dense_projection_i_dim) {
    batch_size = _batch_size, num_channels = _num_channels, i_dim = _i_dim,
    projected_num_channels = _projected_num_channels,
    dense_projection_i_dim = _dense_projection_i_dim;
  }
};

ConcatTestMeta get_test_meta(const std::string file_path) {
  std::fstream myfile(file_path, std::ios_base::in);
  int batch_size, i_dim, num_channels, projected_num_channels,
      dense_projection_i_dim;
  myfile >> batch_size >> i_dim >> num_channels >> projected_num_channels >>
      dense_projection_i_dim;
  return ConcatTestMeta(batch_size,
                        i_dim,
                        num_channels,
                        projected_num_channels,
                        dense_projection_i_dim);
}

void top_level_task(Task const *task,
                    std::vector<PhysicalRegion> const &regions,
                    Context ctx,
                    Runtime *runtime) {
  std::cout << "test framework launched" << std::endl;
  auto test_meta = get_test_meta("test_meta.txt");
  FFConfig ffConfig;
  // create ff model object
  FFModel ff(ffConfig);

  // create embeddings
  int dense_embedding_channels = test_meta.num_channels / 2;
  int sparse_embedding_channels =
      test_meta.num_channels - dense_embedding_channels;
  auto dense_embedding_file_path = "test_input2.txt";
  auto sparse_embedding_file_path = "test_input3.txt";
  Tensor dense_embeddings[dense_embedding_channels];
  Tensor sparse_embeddings[sparse_embedding_channels];
  for (int i = 0; i < dense_embedding_channels; i++) {
    int const dims[2] = {test_meta.batch_size, test_meta.i_dim};
    dense_embeddings[i] = ff.create_tensor<2>(dims, "", DT_FLOAT);
    // init tensor is checked, nothing wrong in init tensor
    // dense_embeddings[i] also checked, it's correct
    initialize_tensor_from_file(
        dense_embedding_file_path, dense_embeddings[i], ff, "float", 2);
  }

  for (int i = 0; i < sparse_embedding_channels; i++) {
    int const dims[2] = {test_meta.batch_size, test_meta.i_dim};
    sparse_embeddings[i] = ff.create_tensor<2>(dims, "", DT_FLOAT);
    // init tensor is checked, nothing wrong in init tensor
    // sparse_embeddings[i] also checked, it's correct
    initialize_tensor_from_file(
        sparse_embedding_file_path, sparse_embeddings[i], ff, "float", 2);
    // std::ostringstream stringStream;
    // stringStream << "sparse_embedding" << i << "_output.txt";
    // std::string copyOfStr = stringStream.str();
    // dump_region_to_file(ff, sparse_embeddings[i].region, copyOfStr, 2);
  }

  // merge two embedding lists
  std::vector<Tensor> dense_embeddings_v(
      dense_embeddings, dense_embeddings + dense_embedding_channels);
  std::vector<Tensor> sparse_embeddings_v(
      sparse_embeddings, sparse_embeddings + sparse_embedding_channels);
  std::vector<Tensor> embeddings;
  embeddings.insert(embeddings.begin(),
                    sparse_embeddings_v.begin(),
                    sparse_embeddings_v.end());
  embeddings.insert(
      embeddings.end(), dense_embeddings_v.begin(), dense_embeddings_v.end());

  auto ret =
      ff.concat("concat_input", test_meta.num_channels, &embeddings[0], 1);

  // load inputs tensors and output gradients tensors for testing
  // use output for output grad (testing only)
  auto output_grad_file_path = "test_output_grad.txt";
  initialize_tensor_gradient_from_file(
      output_grad_file_path, ret, ff, "float", 2);

  ff.optimizer = new SGDOptimizer(&ff, 0.01f);
  // run forward and backward to produce results
  ff.init_layers();
  ff.forward();
  // dump results to file for python validation
  dump_region_to_file(ff, ret.region, "output.txt", 2);
}

void register_custom_tasks() {
  {
    TaskVariantRegistrar registrar(INIT_TENSOR_1D_FROM_FILE_CPU_TASK,
                                   "Load 1d Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<initialize_tensor_from_file_task<1>>(
        registrar, "Load 1d tensor Task");
  }
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
    TaskVariantRegistrar registrar(DUMP_TENSOR_1D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<1>>(registrar,
                                                           "Dump Tensor Task");
  }
  {
    TaskVariantRegistrar registrar(DUMP_TENSOR_2D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<2>>(registrar,
                                                           "Dump Tensor Task");
  }
  {
    TaskVariantRegistrar registrar(DUMP_TENSOR_4D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<4>>(registrar,
                                                           "Dump Tensor Task");
  }
  {
    TaskVariantRegistrar registrar(DUMP_TENSOR_3D_CPU_TASK, "Dump Tensor");
    registrar.add_constraint(ProcessorConstraint(Processor::LOC_PROC));
    registrar.set_leaf();
    Runtime::preregister_task_variant<dump_tensor_task<3>>(registrar,
                                                           "Dump Tensor Task");
  }
}
