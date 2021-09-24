#include "flexflow/tensor.h"
#include <queue>
#include "flexflow/model.h"
#include "flexflow/ops/linear.h"
#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/pool_2d.h"
#include "flexflow/ops/embedding.h"
#include "flexflow/ops/flat.h"
#include "flexflow/ops/element_unary.h"
#include "flexflow/ops/attention.h"
#include "flexflow/ops/element_binary.h"
#include "flexflow/ops/softmax.h"
#include "flexflow/ops/split.h"
#include "flexflow/ops/noop.h"
#include "flexflow/ops/concat.h"
#include "flexflow/utils/hash_utils.h"

namespace FlexFlow {

using namespace Legion;

bool ParallelTensorShape::is_valid() const {
  bool used[MAX_TENSOR_DIM];
  std::fill_n(used, MAX_TENSOR_DIM, false);

  for (int i = 0; i < this->num_dims; i++) {
    ParallelDim const &dim = this->dims[i];
    assert (dim.size > 0);
    assert (dim.degree != ParallelDim::UNKNOWN_DEGREE);
    assert (dim.degree >= 1);
    assert (dim.parallel_idx != ParallelDim::UNKNOWN_INDEX);
    assert (dim.parallel_idx < MAX_TENSOR_DIM);
    used[dims[i].parallel_idx] = true;
    if (dim.size % dim.degree != 0) {
      return false;
    }
  }
  int idx = 0;
  while (used[idx]) {
    idx++;
  }
  for (int i = idx; i < MAX_TENSOR_DIM; i++) {
    assert (!used[i]);
  }

  return true;
}

bool ParallelTensorShape::operator==(const ParallelTensorShape& other) const {
  if (this->num_dims != other.num_dims) {
    return false;
  }

  if (this->data_type != other.data_type) {
    return false;
  }

  for (int i = 0; i < this->num_dims; i++) { 
    if (this->dims[i].size != other.dims[i].size) {
      return false;
    }

    if (this->dims[i].degree != other.dims[i].degree) { 
      return false;
    }
  }


  return true;
}

bool ParallelTensorShape::operator!=(const ParallelTensorShape& other) const {
  return !(*this == other);
}

size_t ParallelTensorShape::get_piece_size() const {
  size_t piece_size = data_type_size(this->data_type);
  for (int i = 0; i < this->num_dims; i++) {
    piece_size *= this->dims[i].size / this->dims[i].degree;
  }
  return piece_size;
}

std::unordered_map<int, int> ParallelTensorShape::get_mv_dim_to_tensor_dim_mapping() const {
  std::unordered_map<int, int> result;
  for (int i = 0; i < this->num_dims; i++) {
    int machine_view_dim = this->dims[i].parallel_idx;
    if (machine_view_dim != -1) {
      assert (result.find(machine_view_dim) == result.end());
      result[machine_view_dim] = i;
    }
  }
  return result;
}

std::unordered_map<int, int> ParallelTensorShape::get_tensor_dim_to_mv_dim_mapping() const {
  std::unordered_map<int, int> result;
  for (auto const &kv : this->get_mv_dim_to_tensor_dim_mapping()) {
    assert (result.find(kv.second) == result.end());
    result[kv.second] = kv.first;
  }
  return result;
}

bool ParallelTensorBase::update_parallel_ids(
    int numdim,
    ParallelDim* dims)
{
  int next_parallel_idx = 0; 
  for (int i = 0; i < numdim; i++) {
    if (dims[i].degree == 1) {
      dims[i].parallel_idx = -1;
    } else {
      dims[i].parallel_idx = next_parallel_idx;
      next_parallel_idx++;
    }
  }
  
  return true;
}

ParallelTensorBase::ParallelTensorBase(const ParallelTensorBase& rhs)
{
  ts_guid = rhs.ts_guid;
  num_dims = rhs.num_dims;
  for (int i = 0; i < num_dims; i++)
    dims[i] = rhs.dims[i];
  machine_view = rhs.machine_view;
  parallel_is = rhs.parallel_is;
  region = rhs.region;
  region_grad = rhs.region_grad;
  part = rhs.part;
  part_grad = rhs.part_grad;
  owner_op = rhs.owner_op;
  owner_idx = rhs.owner_idx;
  data_type = rhs.data_type;
  sync_type = rhs.sync_type;
  initializer = rhs.initializer;
  create_gradients = rhs.create_gradients;
}

void ParallelTensorBase::inline_map(FFConfig &config)
{
  printf("inline map tensor\n");
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;

  RegionRequirement region_req(region, READ_WRITE, EXCLUSIVE, region);
  region_req.add_field(FID_DATA);
  InlineLauncher inline_launcher(region_req);
  physical_region = runtime->map_region(ctx, inline_launcher);
  physical_region.wait_until_valid();
}

void ParallelTensorBase::inline_unmap(FFConfig &config)
{
  printf("inline unmap tensor\n");
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  assert(physical_region.is_valid() == true);
  runtime->unmap_region(ctx, physical_region);
}

template<typename T>
T* ParallelTensorBase::get_raw_ptr(FFConfig &config)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  RegionRequirement region_req(region, READ_WRITE, EXCLUSIVE, region);
  region_req.add_field(FID_DATA);
  T *raw_ptr = NULL;
  if (num_dims == 1) {
    TensorAccessorW<T, 1> acc(physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T*)acc.ptr;
  } else if (num_dims == 2) {
    TensorAccessorW<T, 2> acc(physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T*)acc.ptr;
  } else if (num_dims == 3) {
    TensorAccessorW<T, 3> acc(physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T*)acc.ptr;
  } else if (num_dims == 4) {
    TensorAccessorW<T, 4> acc(physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T*)acc.ptr;
  } else {
    printf("wrong num_dims %d", num_dims);
    assert(0);
  }
  return raw_ptr;
}

void ParallelTensorBase::attach_raw_ptr(FFConfig &config, void *raw_ptr, bool column_major)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  AttachLauncher launcher(EXTERNAL_INSTANCE, region, region);
  std::vector<FieldID> fields(1, FID_DATA);
  const Memory local_sysmem = Machine::MemoryQuery(Machine::get_machine())
       .has_affinity_to(runtime->get_executing_processor(ctx))
       .only_kind(Memory::SYSTEM_MEM)
       .first();
  launcher.attach_array_soa(raw_ptr, column_major,
                            fields, local_sysmem);
  physical_region = runtime->attach_external_resource(ctx, launcher);
}

void ParallelTensorBase::detach_raw_ptr(FFConfig &config)
{
  Context ctx = config.lg_ctx;
  Runtime* runtime = config.lg_hlr;
  runtime->detach_external_resource(ctx, physical_region);
}

template <typename T>
bool ParallelTensorBase::get_input_sub_tensor_via_mappings(const ParallelConfig& pc, ParallelTensorBase& tensor) const
{
  if (pc.nDims != num_dims) {
    printf("Could not get input subtensor because the number of dimensions do not match: %d != %d\n", pc.nDims, num_dims);
    return false;
  }
  std::vector<ParallelDimMappingRecord> mapping;
  T::construct_output_mappings(mapping);
  std::unordered_map<int, int> dim_mapping = input_to_output_mapping(mapping);

  for (int i = 0; i < this->num_dims; i++) {
    assert(pc.dim[dim_mapping.at(i)] == dims[i].degree);
    tensor.dims[i].size = dims[i].size / dims[i].degree;
  }

  return true;
}

bool ParallelTensorBase::get_input_sub_tensor(
    const ParallelConfig& pc,
    ParallelTensorBase& tensor,
    OperatorType type)
{
  //TODO: consider reduction dim for conv2d and linear
  switch (type) {
    case OP_FLAT:
      {
        assert (pc.nDims == 3 && "Invalid dimension for parallel config of OP_FLAT");

        tensor.num_dims = this->num_dims;
        for (int i = 0; i < 3; i++) {
          assert (tensor.dims[i].size % pc.dim[i] == 0);
          tensor.dims[i].size = tensor.dims[i].size / pc.dim[i];
        }
        break;
      }
    case OP_RESHAPE:
      {
        for (int i = 0; i < pc.nDims - 1; i ++)
          assert(pc.dim[i] == 1 && "Assuming data parallel for RESHAPE");
        int batchDim = pc.dim[pc.nDims-1];
        if (dims[num_dims-1].size % batchDim != 0) {
          printf("Could not get input subtensor because the dimension is not divisiable: %d %% %d != 0\n", dims[num_dims-1].size, batchDim);
        }
        tensor.num_dims = num_dims;
        for (int i = num_dims-2; i >= 0; i--) {
          tensor.dims[i].size = dims[i].size;
        }
        tensor.dims[num_dims-1].size = dims[num_dims-1].size / batchDim;
        break;
      }
    case OP_LINEAR:
      {
        if (pc.nDims != num_dims) {
          printf("Could not get input subtensor because the number of dimensions do not match: %d != %d\n", pc.nDims, num_dims);
          return false;
        }
        tensor.num_dims = num_dims;
        for (int i = 0; i < num_dims; i++) {
          if (dims[i].size % pc.dim[i] != 0) {
            printf("Could not get input subtensor because the given dimension is not divisible: %d %% %d != 0\n", dims[i].size, pc.dim[i]);
            return false;
          }
          tensor.dims[i].size = dims[i].size / pc.dim[i];
        }
        tensor.dims[0].size = dims[0].size;
        tensor.data_type = data_type;
	break;
      }
    case OP_CONV2D:
      if (!this->get_input_sub_tensor_via_mappings<Conv2D>(pc, tensor)) {
        return false;
      }
      break;
    case OP_POOL2D:
      if (!this->get_input_sub_tensor_via_mappings<Pool2D>(pc, tensor)) {
        return false;
      }
      break;
    default:
      {
        if (pc.nDims != num_dims) {
          printf("Could not get input subtensor because the number of dimensions do not match: %d != %d\n", pc.nDims, num_dims);
          return false;
        }
        for (int i = 0; i < num_dims; i++) {
          if (dims[i].size % pc.dim[i] != 0) {
            printf("Could not get input subtensor because the given dimension is not divisible: %d %% %d != 0\n", dims[i].size, pc.dim[i]);
            return false;
          }
        }
        tensor.num_dims = num_dims;
        for (int i = 0; i < num_dims; i++) {
          tensor.dims[i].size = dims[i].size / pc.dim[i];
        }
        tensor.data_type = data_type;
      }
      break;
  }
  return true;
}

bool ParallelTensorBase::get_output_sub_tensor(
    const ParallelConfig& pc,
    ParallelTensorBase& tensor,
    OperatorType type)
{
  if (pc.nDims != num_dims) {
    printf("Could not get output subtensor because the number of dimensions do not match: %d != %d\n", pc.nDims, num_dims);
    return false;
  }
  for (int i = 0; i < num_dims; i++) {
    if (dims[i].size % pc.dim[i] != 0) {
      printf("Could not get output subtensor because the given dimension is not divisible: %d %% %d != 0\n", dims[i].size, pc.dim[i]);
      return false;
    }
  }
  tensor.num_dims = num_dims;
  for (int i = 0; i < num_dims; i++)
    tensor.dims[i].size = dims[i].size / pc.dim[i];
  tensor.data_type = data_type;
  return true;
}

size_t ParallelTensorBase::get_owner_independent_hash() const
{
  size_t hash = 17 * 31 + std::hash<int>()((int)data_type);
  hash = hash * 31 + std::hash<int>()((int)sync_type);
  hash = hash * 31 + std::hash<int>()(num_dims);
  for (int i = 0; i < num_dims; i++) {
    hash = hash * 31 + std::hash<int>()(dims[i].size);
    hash = hash * 31 + std::hash<int>()(dims[i].degree);
    hash = hash * 31 + std::hash<int>()(dims[i].parallel_idx);
  }
  return hash;
}

size_t ParallelTensorBase::get_volume() const
{
  size_t volume = 1;
  for (int i = 0; i < num_dims; i++)
    volume *= dims[i].size;
  return volume;
}

size_t ParallelTensorBase::get_total_num_parts() const
{
  size_t parts = 1;
  for (int i = 0; i < num_dims; i++)
    parts *= dims[i].degree;
  return parts;
}

Domain ParallelTensorBase::get_domain() const
{
  Domain d;
  d.dim = this->num_dims;
  for (int i = 0; i < this->num_dims; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i+d.dim] = this->dims[i].size - 1;
  }
  return d;
}

bool ParallelTensorBase::check_valid() const
{
  bool used[MAX_TENSOR_DIM];
  for (int i = 0; i < MAX_TENSOR_DIM; i++)
    used[i] = false;
  for (int i = 0; i < num_dims; i++) {
    if (dims[i].size < 0) 
      return false;
    if (dims[i].size % dims[i].degree != 0)
      return false;
    if (dims[i].parallel_idx > MAX_TENSOR_DIM)
      return false;
    assert (dims[i].parallel_idx >= -1);
    assert (dims[i].degree >= 1);
    if (dims[i].parallel_idx >= 0) {
      if (used[dims[i].parallel_idx])
        return false;
      used[dims[i].parallel_idx] = true;
    }
  }
  assert (this->data_type != DT_NONE);
  int idx = 0;
  while (used[idx]) idx++;
  for (int i = idx; i < MAX_TENSOR_DIM; i++)
    if (used[i]) return false;
  return true;
}

void ParallelTensorBase::print(const std::string& name) const
{
  printf("%s: sizes[", name.c_str());

  for (int i = 0; i < num_dims; i++) {
    printf("%d ", dims[i].size);
  }
  printf("] degree[");
  for (int i = 0; i < num_dims; i++)
    printf("%d ", dims[i].degree);
  printf("] parallel_ids[");
  for (int i = 0; i < num_dims; i++)
    printf("%d ", dims[i].parallel_idx);
  printf("]\n");

}

ParallelTensorShape ParallelTensorBase::get_shape() const {
  ParallelTensorShape shape;
  shape.num_dims = this->num_dims;
  shape.data_type = this->data_type;
  for (int i = 0; i < this->num_dims; i++) {
    shape.dims[i] = this->dims[i];
  }

  return shape;
}

std::ostream& operator<<(std::ostream &s, ParallelTensorShape const &shape) {
  s << "[ ";
  for (int i = 0; i < shape.num_dims; i++) {
    s << shape.dims[i].size << "/" << shape.dims[i].degree << " ";
  }
  s << "]";

  return s;
}

}; // namespace FlexFlow

namespace std {
  size_t hash<FlexFlow::ParallelTensorShape>::operator()(FlexFlow::ParallelTensorShape const &shape) const {
    size_t key = 0;
    hash_combine(key, shape.num_dims);
    for (int i = 0; i < shape.num_dims; i++) {
      hash_combine(key, shape.dims[i].size);
      hash_combine(key, shape.dims[i].degree);
    }
    return key;
  }
};

namespace FlexFlow {

bool ParallelTensorBase::is_valid_machine_view(const MachineView& view) const
{
  int is_dim = 0;
  for (int i = 0; i < num_dims; i++)
    if (dims[i].parallel_idx != -1) {
      is_dim++;
      if (dims[i].parallel_idx > view.ndims)
        return false;
      if (view.dim[dims[i].parallel_idx] != dims[i].degree)
        return false;
    }
  if (is_dim == 0) {
    is_dim = 1;
  }
  if (is_dim != view.ndims)
    return false;
  if (get_total_num_parts() != view.num_parts())
    return false;
  return true;
}

template float* ParallelTensorBase::get_raw_ptr<float>(FFConfig &config);
template int32_t* ParallelTensorBase::get_raw_ptr<int32_t>(FFConfig &config);

}; // namespace FlexFlow
