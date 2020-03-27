#include "strategy.pb.h"
#include <fstream>
#include <iostream>


class FFStrategy {
public:
  FFStrategy(int _gpus_per_node, int _num_nodes);
  bool add_conv_config(const std::string& name,
                       const std::string& device_type,
                       int num_par_n,
                       int num_par_c,
                       const std::string& input_memory,
                       const std::string& output_memory);
  FFProtoBuf::Op_DeviceType to_device_type(const std::string& name) {
    if (name == "gpu" || name == "GPU") {
      return FFProtoBuf::Op_DeviceType_GPU;
    } else if (name == "cpu" || name == "CPU") {
      return FFProtoBuf::Op_DeviceType_CPU;
    } else {
      assert(false);
    }
    return FFProtoBuf::Op_DeviceType_GPU;
  }

  FFProtoBuf::Op_MemoryType to_memory_type(const std::string& name) {
    if (name == "gpu_memory" || name == "FBM") {
      return FFProtoBuf::Op_MemoryType_FBM;
    } else if (name == "cpu_memory" || name == "ZCM") {
      return FFProtoBuf::Op_MemoryType_ZCM;
    } else {
      assert(false);
    }
    return FFProtoBuf::Op_MemoryType_FBM;
  }
  bool add_embed_config(const std::string& name,
                        const std::string& device_type,
                        const std::string& input_memory_type,
                        const std::string& weight_memory_type,
                        const std::string& output_memory_type,
                        int gpu_id);
  bool add_concat_config(const std::string& name,
                         const std::string& device_type,
                         const std::string& input_memory_type,
                         const std::string& output_memory_type,
                         int num_parts_sample,
                         const std::vector<int>& device_ids);
  bool add_linear_config(const std::string& name,
                         const std::string& device_type,
                         const std::string& input_memory_type,
                         const std::string& weight_memory_type,
                         const std::string& output_memory_type,
                         int num_parts_channel,
                         int num_parts_sample,
                         const std::vector<int>& device_ids);
  bool add_mse_config(const std::string& name,
                      const std::string& device_type,
                      const std::string& input_memory_type,
                      int num_parts_batch,
                      const std::vector<int>& device_ids);
  void export_file(const std::string& file);
private:
  int gpus_per_node, num_nodes;
  FFProtoBuf::Strategy strategy;
};

FFStrategy::FFStrategy(int _gpus_per_node, int _num_nodes)
: gpus_per_node(_gpus_per_node), num_nodes(_num_nodes)
{
  if (_gpus_per_node <= 0 || _num_nodes <= 0) {
    printf("Invalide input configurations...\n");
    exit(0);
  }
}

bool FFStrategy::add_embed_config(const std::string& name,
                                  const std::string& device_type,
                                  const std::string& input_memory_type,
                                  const std::string& weight_memory_type,
                                  const std::string& output_memory_type,
                                  int gpu_id)
{
  FFProtoBuf::Op* op = strategy.add_ops();
  op->set_name(name);
  op->set_device_type(to_device_type(device_type));
  op->add_memory_types(to_memory_type(input_memory_type));
  op->add_memory_types(to_memory_type(weight_memory_type));
  op->add_memory_types(to_memory_type(output_memory_type));
  op->add_dims(1);
  op->add_dims(1);
  for (int j = 0; j < 1; j++) {
    op->add_device_ids(gpu_id);
  }
  return true;
}

bool FFStrategy::add_concat_config(const std::string& name,
                                   const std::string& device_type,
                                   const std::string& input_memory_type,
                                   const std::string& output_memory_type,
                                   int num_parts_sample,
                                   const std::vector<int>& device_ids)
{
  FFProtoBuf::Op* op = strategy.add_ops();
  op->set_name(name);
  op->set_device_type(to_device_type(device_type));
  op->add_memory_types(to_memory_type(input_memory_type));
  op->add_memory_types(to_memory_type(output_memory_type));
  op->add_dims(1);
  op->add_dims(num_parts_sample);
  assert(num_parts_sample == (int) device_ids.size());
  for (int i = 0; i < num_parts_sample; i++)
    op->add_device_ids(device_ids[i]);
}

bool FFStrategy::add_linear_config(const std::string& name,
                                   const std::string& device_type,
                                   const std::string& input_memory_type,
                                   const std::string& weight_memory_type,
                                   const std::string& output_memory_type,
                                   int num_parts_channel,
                                   int num_parts_sample,
                                   const std::vector<int>& device_ids)
{
  FFProtoBuf::Op* op = strategy.add_ops();
  op->set_name(name);
  op->set_device_type(to_device_type(device_type));
  op->add_memory_types(to_memory_type(input_memory_type));
  op->add_memory_types(to_memory_type(weight_memory_type));
  op->add_memory_types(to_memory_type(output_memory_type));
  op->add_dims(num_parts_channel);
  op->add_dims(num_parts_sample);
  assert(num_parts_sample * num_parts_channel == (int) device_ids.size());
  for (int i = 0; i < num_parts_channel * num_parts_sample; i++)
    op->add_device_ids(device_ids[i]);
}

bool FFStrategy::add_mse_config(const std::string& name,
                                const std::string& device_type,
                                const std::string& input_memory_type,
                                int num_parts_sample,
                                const std::vector<int>& device_ids)
{
  FFProtoBuf::Op* op = strategy.add_ops();
  op->set_name(name);
  op->set_device_type(to_device_type(device_type));
  op->add_memory_types(to_memory_type(input_memory_type));
  op->add_dims(1);
  op->add_dims(num_parts_sample);
  assert(num_parts_sample == (int) device_ids.size());
  for (int i = 0; i < num_parts_sample; i++)
    op->add_device_ids(device_ids[i]);
}

void FFStrategy::export_file(const std::string& output)
{
  std::fstream outputFile(output.c_str(), std::ios::out | std::ios::trunc);
  strategy.SerializeToOstream(&outputFile);
}

void parse_input_args(char **argv, int argc, int& gpus_per_node, int& num_nodes)
{
  for (int i = 1; i < argc; i++) {
    if (!strcmp(argv[i], "--gpu")) {
      gpus_per_node = std::atoi(argv[++i]);
      continue;
    }
    if (!strcmp(argv[i], "--node")) {
      num_nodes = std::atoi(argv[++i]);
      continue;
    }
  }
}

int main(int argc, char **argv)
{
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  int gpus_per_node = 0, num_nodes = 0;
  parse_input_args(argv, argc, gpus_per_node, num_nodes);
  printf("Number of GPUs Per Node = %d\n", gpus_per_node);
  printf("Number of Nodes = %d\n", num_nodes);
  FFStrategy strategy(gpus_per_node, num_nodes);
  // Embedding
  for (int i = 0; i < 24; i++) {
    std::string name = "embedding"+std::to_string(i);
    strategy.add_embed_config(name, "GPU", "FBM"/*input*/,
        "FBM"/*weight*/, "FBM"/*output*/, i % (gpus_per_node * num_nodes));
  }
  {
    std::vector<int> device_ids;
    for (int i = 0; i < num_nodes; i++)
      device_ids.push_back(i * gpus_per_node);
    strategy.add_concat_config("concat", "GPU", "FBM"/*input*/,
        "FBM"/*output*/, num_nodes, device_ids);
  }
  {
    std::vector<int> device_ids;
    for (int i = 0; i < num_nodes * gpus_per_node; i++)
      device_ids.push_back(i);
    strategy.add_linear_config("linear", "GPU", "FBM"/*input*/, "FBM"/*weight*/,
        "FBM"/*output*/, 1, num_nodes*gpus_per_node, device_ids);
  }
  {
    std::vector<int> device_ids;
    for (int i = 0; i < num_nodes * gpus_per_node; i++)
      device_ids.push_back(i);
    strategy.add_mse_config("mse_loss", "GPU", "FBM"/*input*/,
        num_nodes*gpus_per_node, device_ids);
  }
  std::string output = "dlrm_strategy_gpu_" + std::to_string(gpus_per_node) + "_node_" + std::to_string(num_nodes) + ".pb";
  strategy.export_file(output);
  google::protobuf::ShutdownProtobufLibrary();
}
