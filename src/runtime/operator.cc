#include "flexflow/operator.h"
#include "flexflow/ffconst_utils.h"
#include "flexflow/simulator.h"
#include <stdexcept>

#include <sys/stat.h>
#include <sys/types.h>
#if defined(FF_USE_CUDA) || defined(FF_USE_HIP_CUDA)
#include "flexflow/utils/cuda_helper.h"
#else
#include "flexflow/utils/hip_helper.h"
#endif

namespace FlexFlow {

size_t Op::get_untyped_params_hash() const {
  size_t hash = this->get_params_hash();
  hash_combine(hash, this->op_type);
  return hash;
}

size_t Op::get_params_hash() const {
  throw std::runtime_error(
      "No overload of get_params_hash defined for op type " +
      get_operator_type_name(this->op_type));
}

/*static*/
void Op::save_inference_tensors_to_file(
    OpMeta *m,
    int shard_id,
    BatchConfig const *bc,
    std::vector<GenericTensorAccessorR> input_tensors,
    std::vector<GenericTensorAccessorR> weight_tensors,
    std::vector<GenericTensorAccessorW> output_tensors) {

  // Check if output directory exists, and create it if it does not
  char const *folder_path = "./inference_tensors";
  struct stat st = {0};
  if (stat(folder_path, &st) == -1) {
    // Directory does not exist, create it
    mkdir(folder_path, 0700);
  }
  // output base filepath, shared by all tensors from the same operator
  std::string base_filepath =
      "./inference_tensors/model_" + std::to_string(m->layer_guid.model_id) +
      "_decoding-step_" + std::to_string(m->decoding_step) + "_layer-num_" +
      std::to_string(m->layer_guid.transformer_layer_id) + "_layer-name_" +
      m->op_name + "_shard-id_" + std::to_string(shard_id);
  // save batch config, if passed
  if (bc != nullptr) {
    bc->save_to_file(base_filepath + "_batch-config");
  }
  // save all inputs
  for (int i = 0; i < input_tensors.size(); i++) {
    std::string filename = base_filepath + "_input_" + std::to_string(i);
    if (input_tensors[i].data_type == DT_FLOAT) {
      save_tensor(input_tensors[i].get_float_ptr(),
                  input_tensors[i].domain.get_volume(),
                  filename.c_str());
    } else if (input_tensors[i].data_type == DT_HALF) {
      save_tensor(input_tensors[i].get_half_ptr(),
                  input_tensors[i].domain.get_volume(),
                  filename.c_str());
    } else if (input_tensors[i].data_type == DT_INT32) {
      save_tensor(input_tensors[i].get_int32_ptr(),
                  input_tensors[i].domain.get_volume(),
                  filename.c_str());
    } else if (input_tensors[i].data_type == DT_INT64) {
      save_tensor(input_tensors[i].get_int64_ptr(),
                  input_tensors[i].domain.get_volume(),
                  filename.c_str());
    } else {
      assert(false && "Tensor data type not supported");
    }
  }
  // only dump the weights once
  if (m->decoding_step == 0) {
    for (int i = 0; i < weight_tensors.size(); i++) {
      std::string filename = base_filepath + "_weight_" + std::to_string(i);
      if (weight_tensors[i].data_type == DT_FLOAT) {
        save_tensor(weight_tensors[i].get_float_ptr(),
                    weight_tensors[i].domain.get_volume(),
                    filename.c_str());
      } else if (weight_tensors[i].data_type == DT_HALF) {
        save_tensor(weight_tensors[i].get_half_ptr(),
                    weight_tensors[i].domain.get_volume(),
                    filename.c_str());
      } else if (weight_tensors[i].data_type == DT_INT32) {
        save_tensor(weight_tensors[i].get_int32_ptr(),
                    weight_tensors[i].domain.get_volume(),
                    filename.c_str());
      } else if (weight_tensors[i].data_type == DT_INT64) {
        save_tensor(weight_tensors[i].get_int64_ptr(),
                    weight_tensors[i].domain.get_volume(),
                    filename.c_str());
      } else {
        assert(false && "Tensor data type not supported");
      }
    }
  }
  // save all outputs
  for (int i = 0; i < output_tensors.size(); i++) {
    std::string filename = base_filepath + "_output_" + std::to_string(i);
    if (output_tensors[i].data_type == DT_FLOAT) {
      save_tensor(output_tensors[i].get_float_ptr(),
                  output_tensors[i].domain.get_volume(),
                  filename.c_str());
    } else if (output_tensors[i].data_type == DT_HALF) {
      save_tensor(output_tensors[i].get_half_ptr(),
                  output_tensors[i].domain.get_volume(),
                  filename.c_str());
    } else if (output_tensors[i].data_type == DT_INT32) {
      save_tensor(output_tensors[i].get_int32_ptr(),
                  output_tensors[i].domain.get_volume(),
                  filename.c_str());
    } else if (output_tensors[i].data_type == DT_INT64) {
      save_tensor(output_tensors[i].get_int64_ptr(),
                  output_tensors[i].domain.get_volume(),
                  filename.c_str());
    } else {
      assert(false && "Tensor data type not supported");
    }
  }
  // increase count of decoding steps
  m->decoding_step++;
}

}; // namespace FlexFlow