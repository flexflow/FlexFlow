#ifndef _FF_UTILS_H_
#define _FF_UTILS_H_
#include "model.h"
#include <sstream>
#include <fstream>
#include <iomanip>
#include <iostream>

struct ArgsConfig;

void initialize_tensor_from_file(const std::string file_path, 
  Tensor label, 
  const FFModel& ff, 
  std::string data_type="float", 
  int num_dim=3);

void initialize_tensor_gradient_from_file(const std::string file_path, 
  Tensor label, 
  const FFModel& ff, 
  std::string data_type,  int num_dim);

void initialize_tensor_from_file(const std::string file_path, 
  Tensor label, 
  const FFModel& ff, 
  std::string data_type,  int num_dim);

template<int DIM>
void initialize_tensor_from_file_task(const Task *task,
  const std::vector<PhysicalRegion> &regions,
  Context ctx,
  Runtime* runtime);

void dump_region_to_file(FFModel &ff, 
  LogicalRegion &region, 
  std::string file_path, 
  int dims=4);       
     
template<int DIM>
void dump_tensor_task(const Task* task,
  const std::vector<PhysicalRegion>& regions,
  Context ctx, Runtime* runtime);     

void register_custom_tasks();
#endif