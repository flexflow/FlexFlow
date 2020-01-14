#include "strategy.pb.h"
#include <fstream>
#include <iostream>


class FFStrategy {
public:
  FFStrategy(void);
  bool add_conv_config(const std::string& name,
                       const std::string& device_type,
                       int num_par_n,
                       int num_par_c,
                       const std::string& input_memory,
                       const std::string& output_memory);
private:
  FFProtoBuf::Strategy strategy;
}

FFStrategy::FFStrategy(void)
{}

bool FFStrategy::add_conv_config(const std::string& name,
                                 const std::string& device_type,
                                 int num_par_n,
                                 int num_par_c)
{
}

int main()
{
  int gpu = 6;
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  FFProtoBuf::Strategy strategy;
  // Embedding
  for (int i = 0; i < 24; i++) {
    std::string name = "embedding"+std::to_string(i);
    FFProtoBuf::Op* op = strategy.add_ops();
    op->set_name(name);
    op->set_device_type(FFProtoBuf::Op_DeviceType_GPU);
    op->add_dims(1);
    op->add_dims(1);
    for (int j = 0; j < 1; j++)
      op->add_device_ids(i % gpu);
  }
  if (false) {
    FFProtoBuf::Op* op = strategy.add_ops();
    op->set_name("concat");
    op->set_device_type(FFProtoBuf::Op_DeviceType_GPU);
    op->add_dims(1);
    op->add_dims(gpu / 6);
    for (int j = 0; j < gpu / 6; j++)
      op->add_device_ids(j * 6);
  }
  std::vector<std::string> names;
  names.push_back("concat");
  names.push_back("linear");
  names.push_back("mse_loss");
  for (size_t i = 0; i < names.size(); i++) {
    FFProtoBuf::Op* op = strategy.add_ops();
    op->set_name(names[i]);
    op->set_device_type(FFProtoBuf::Op_DeviceType_GPU);
    op->add_dims(1);
    op->add_dims(gpu);
    for (int j = 0; j < gpu; j++)
      op->add_device_ids(j);
  }
  std::string output = "dlrm_strategy_" + std::to_string(gpu) + "gpus.pb";
  std::fstream outputFile(output.c_str(), std::ios::out | std::ios::trunc);
  strategy.SerializeToOstream(&outputFile);
  google::protobuf::ShutdownProtobufLibrary();
}
