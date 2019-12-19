#include "strategy.pb.h"
#include <fstream>
#include <iostream>

int main()
{
  int gpu = 1;
  int cpu = 1;
  int nemb = 8; //Assuming >gpu embeddings 1x per GPU and the rest distributed among available CPUs

  GOOGLE_PROTOBUF_VERIFY_VERSION;
  FFProtoBuf::Strategy strategy;
  
  // Embedding
  int ei = 0;
#if 0
  for (ei = 0; ei < std::min(gpu, nemb); ei++) {
    std::string name = "embedding"+std::to_string(ei);
    FFProtoBuf::Op* op = strategy.add_ops();
    op->set_name(name);
    op->set_device_type(FFProtoBuf::Op_DeviceType_GPU);
    op->add_dims(1);
    op->add_dims(1);
    op->add_device_ids(ei);
  }
#endif

  for (;ei < nemb; ei++) {
    std::string name = "embedding"+std::to_string(ei);
    FFProtoBuf::Op* op = strategy.add_ops();
    op->set_name(name);
    op->set_device_type(FFProtoBuf::Op_DeviceType_CPU);
    op->add_dims(1);
    op->add_dims(1);
    op->add_device_ids(ei%cpu);
  }
  std::vector<std::string> names;
  names.push_back("linear");
  names.push_back("mse_loss");
  names.push_back("concat");
  for (size_t i = 0; i < names.size(); i++) {
    FFProtoBuf::Op* op = strategy.add_ops();
    op->set_name(names[i]);
    op->set_device_type(FFProtoBuf::Op_DeviceType_GPU);
    op->add_dims(1);
    op->add_dims(gpu);
    for (int j = 0; j < gpu; j++)
      op->add_device_ids(j);
  }
  std::string output = "dlrm_strategy_" + std::to_string(nemb) + "nEmb_" + std::to_string(cpu) + "cpu_" + std::to_string(gpu) + "gpu.pb";
  std::fstream outputFile(output.c_str(), std::ios::out | std::ios::trunc);
  strategy.SerializeToOstream(&outputFile);
  google::protobuf::ShutdownProtobufLibrary();
}
