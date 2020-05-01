#ifndef __FLEXFLOW_DATALOADER_H__
#define __FLEXFLOW_DATALOADER_H__

#include "model.h"

struct NetConfig {
  NetConfig(void);
  std::string dataset_path;
};

class ImgDataLoader {
public:
  ImgDataLoader();
  static void load_label(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
  void reset(void);             
public:
  int num_samples, next_index;
  Tensor full_input, batch_input;
  Tensor full_label, batch_label;
};

class ImgDataLoader4D : public ImgDataLoader {
public:
  ImgDataLoader4D(FFModel& ff, Tensor input, Tensor label, 
                  Tensor full_input_, Tensor full_label_, int num_samples_);
  ImgDataLoader4D(FFModel& ff, const NetConfig& alexnet, 
                  Tensor input, Tensor label);
  static void load_input(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
  static void load_entire_dataset(const Task *task,
                                  const std::vector<PhysicalRegion> &regions,
                                  Context ctx,
                                  Runtime* runtime);
  static void load_entire_dataset_from_numpy(const Task *task,
                                             const std::vector<PhysicalRegion> &regions,
                                             Context ctx,
                                             Runtime* runtime);
  void next_batch(FFModel&);
private:
  size_t get_file_size(const std::string& filename);              
};

class ImgDataLoader2D : public ImgDataLoader {
public:
  ImgDataLoader2D(FFModel& ff, Tensor input, Tensor label, 
                  Tensor full_input_, Tensor full_label_, int num_samples_);
  static void load_input(const Task *task,
                         const std::vector<PhysicalRegion> &regions,
                         Context ctx,
                         Runtime* runtime);
  static void load_entire_dataset_from_numpy(const Task *task,
                                            const std::vector<PhysicalRegion> &regions,
                                            Context ctx,
                                            Runtime* runtime);
  void next_batch(FFModel&);
};

class SingleDataLoader {
public:
  SingleDataLoader(FFModel& ff, Tensor input, Tensor full_input_, int num_samples_, DataType datatype_);
  
  void next_batch(FFModel&);
  
  void reset(void); 
  
  static void register_cpu_tasks(void);
  
  static void register_gpu_tasks(void);
    
  template<typename DT>
  static void load_input_2d(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx,
                            Runtime* runtime);
  template<typename DT>
  static void load_input_4d(const Task *task,
                            const std::vector<PhysicalRegion> &regions,
                            Context ctx,
                            Runtime* runtime);
  template<typename DT, int NDIM>                                
  static void load_entire_dataset_from_numpy(const Task *task,
                                             const std::vector<PhysicalRegion> &regions,
                                             Context ctx,
                                             Runtime* runtime);
private:
  template<int NDIM>
  void next_batch_xd_launcher(FFModel&, int task_id);
public:
  int num_samples, next_index;
  DataType datatype;
  Tensor full_input, batch_input;
  Tensor full_label, batch_label;             
};

#define MAX_NUM_SAMPLES 4196
struct SampleIdxs {
  int num_samples;
  int idxs[MAX_NUM_SAMPLES];
};

#endif // __FLEXFLOW_DATALOADER_H__