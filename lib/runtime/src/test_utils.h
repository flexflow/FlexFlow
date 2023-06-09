#ifndef _FLEXFLOW_UTILS_H_
#define _FLEXFLOW_UTILS_H_

#include <fstream>
#include <iomanip>
#include <iostream>
#include <sstream>

namespace FlexFlow {

struct ArgsConfig;

void initialize_tensor_from_file(const std::string file_path,
                                 Tensor label,
                                 FFModel const &ff,
                                 std::string data_type = "float",
                                 int num_dim = 3);

void initialize_tensor_gradient_from_file(const std::string file_path,
                                          Tensor label,
                                          FFModel const &ff,
                                          std::string data_type,
                                          int num_dim);

void initialize_tensor_from_file(const std::string file_path,
                                 Tensor label,
                                 FFModel const &ff,
                                 std::string data_type,
                                 int num_dim);

template <int DIM>
void initialize_tensor_from_file_task(
    Legion::Task const *task,
    std::vector<Legion::PhysicalRegion> const &regions,
    Legion::Context ctx,
    Legion::Runtime *runtime);

void dump_region_to_file(FFModel &ff,
                         Legion::LogicalRegion &region,
                         std::string file_path,
                         int dims = 4);

template <int DIM>
void dump_tensor_task(Legion::Task const *task,
                      std::vector<Legion::PhysicalRegion> const &regions,
                      Legion::Context ctx,
                      Legion::Runtime *runtime);

void register_custom_tasks();

}; // namespace FlexFlow
#endif
