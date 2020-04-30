#ifndef __FLEXFLOW_C_INTERNAL_H__
#define __FLEXFLOW_C_INTERNAL_H__

#include "model.h"

struct NetConfig {
  NetConfig(void);
  std::string dataset_path;
};

class ImgDataLoader {
public:
  ImgDataLoader(FFModel& ff, const NetConfig& alexnet, 
                Tensor input, Tensor label, Tensor full_input_, Tensor full_label_);
  ImgDataLoader(FFModel& ff, const NetConfig& alexnet, 
                Tensor input, Tensor label);
  static void load_input(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
  static void load_label(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
  static void load_entire_dataset(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx,
                                  Runtime* runtime);
  void next_batch(FFModel&);
  void reset(void);  
private:
  size_t get_file_size(const std::string& filename);              
public:
  int num_samples, next_index;
  Tensor full_input, batch_input;
  Tensor full_label, batch_label;
};

#define MAX_NUM_SAMPLES 4196
struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};

#endif // __FLEXFLOW_C_INTERNAL_H__