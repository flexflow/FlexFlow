#include "strategy.pb.h"
#include <fstream>
#include <iostream>

int main()
{
  int gpu = 1;
  GOOGLE_PROTOBUF_VERIFY_VERSION;
  FFProtoBuf::Strategy strategy;
  // Embedding
  for (int i = 0; i < gpu; i++) {
    std::string name = "embedding"+std::to_string(i);
    FFProtoBuf::Op* op = strategy.add_ops();
    op->set_name(name);
    op->add_dims(gpu);
    op->add_dims(1);
    for (int j = 0; j < gpu; j++)
      op->add_devices(i);
  }
  std::vector<std::string> names;
  names.push_back("linear");
  names.push_back("mse_loss");
  names.push_back("concat");
  for (size_t i = 0; i < names.size(); i++) {
    FFProtoBuf::Op* op = strategy.add_ops();
    op->set_name(names[i]);
    op->add_dims(gpu);
    op->add_dims(1);
    for (int j = 0; j < gpu; j++)
      op->add_devices(j);
  }
  std::string output = "dlrm_strategy_" + std::to_string(gpu) + "gpus.pb";
  std::fstream outputFile(output.c_str(), std::ios::out | std::ios::trunc);
  strategy.SerializeToOstream(&outputFile);
  google::protobuf::ShutdownProtobufLibrary();
}
