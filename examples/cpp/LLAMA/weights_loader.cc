/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#include <cudnn.h>
#include <iomanip>
#include <iostream>
// #include <torch/script.h>
// #include <torch/torch.h>
#include "llama.h"

void DataLoader::load_weights(Task const *task,
                           std::vector<PhysicalRegion> const &regions,
                           Context ctx,
                           Runtime *runtime) {

  // // load file
  // torch::jit::script::Module mod;
  // try {
  //   std::cout << "start loading " << pretrained_model_path << "\n";
  //   mod = torch::jit::load(pretrained_model_path);
  //   std::cout << "reading from " << pretrained_model_path << "\n";
  // } catch (torch::Error const &error) {
  //   std::cerr << "error loading pt file " << pretrained_model_path << ".\n";
  //   std::cout << error.msg() << std::endl;
  //   return;
  // }
  // // extract weights

  // for (auto &v : params_map) {
  //   std::string weight_name = v.first;
  //   std::cout << weight_name << std::endl;
  //   // load the tensor from module
  //   auto tensor_loaded = mod.attr(weight_name).toTensor();
  //   // convert tensor to a typed tensor (float used here)
  //   torch::Tensor tensor_loaded_typed = tensor_loaded.to(torch::kFloat32);

  //   long long total_dim = 1;
  //   int num_dim = tensor_loaded_typed.dim();
  //   std::cout << tensor_loaded_typed.dim() << std::endl;
  //   std::cout << "print size of each dim" << std::endl;

  //   for (int i = 0; i < num_dim; i++) {
  //     std::cout << tensor_loaded_typed.size(i) << std::endl;
  //     total_dim *= tensor_loaded_typed.size(i);
  //   }

  //   // convert tensor to cpp array with a specific type (float used here)
  //   float *cpp_array_gpu = tensor_loaded_typed.data_ptr<float>();
  //   // verify correctness of cpp array against tensor
  //   TORCH_CHECK(tensor_loaded_typed.data_ptr() == cpp_array);

  //   // copy to cpu memory and map with model
  //   float *cpp_array_cpu = (float *)malloc(total_dim * sizeof(float));
  //   cudaMemcpy(cpp_array_cpu,
  //              cpp_array_gpu,
  //              total_dim * sizeof(float),
  //              cudaMemcpyDeviceToHost);
  //   v.second = cpp_array_cpu;
  //   cudaFree(cpp_array_gpu);
    
  //   std::cout << "loaded " << weight_name << "\n";
  //   std::cout << "..."
  //             << "\n";
  }
