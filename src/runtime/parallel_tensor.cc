#include "flexflow/ffconst_utils.h"
#include "flexflow/model.h"
#include "flexflow/ops/attention.h"
#include "flexflow/ops/concat.h"
#include "flexflow/ops/conv_2d.h"
#include "flexflow/ops/element_binary.h"
#include "flexflow/ops/element_unary.h"
#include "flexflow/ops/embedding.h"
#include "flexflow/ops/flat.h"
#include "flexflow/ops/linear.h"
#include "flexflow/ops/noop.h"
#include "flexflow/ops/pool_2d.h"
#include "flexflow/ops/softmax.h"
#include "flexflow/ops/split.h"
#include "flexflow/tensor.h"
#include "flexflow/utils/hash_utils.h"
#include <queue>

namespace FlexFlow {

using namespace Legion;

TensorBase::TensorBase(TensorBase const &rhs) {
  tensor_guid = rhs.tensor_guid;
  num_dims = rhs.num_dims;
  for (int i = 0; i < num_dims; i++) {
    dims[i] = rhs.dims[i];
  }
  data_type = rhs.data_type;
  sync_type = rhs.sync_type;
  initializer = rhs.initializer;
  parallel_tensor = rhs.parallel_tensor;
  owner_layer = rhs.owner_layer;
  owner_idx = rhs.owner_idx;
  create_gradients = rhs.create_gradients;
}

size_t TensorBase::get_volume() const {
  size_t volume = 1;
  for (int i = 0; i < num_dims; i++) {
    volume *= dims[i];
  }
  return volume;
}

template <typename T>
bool TensorBase::set_tensor(FFModel const *ff,
                            std::vector<int> const &dim_sizes,
                            T const *data) {
  if (num_dims != (int)dim_sizes.size()) {
    return false;
  }
  for (int i = 0; i < num_dims; i++) {
    if (dims[num_dims - 1 - i] != dim_sizes[i]) {
      return false;
    }
  }
  ParallelTensor ptensor = nullptr;
  ff->get_parallel_tensor_from_tensor(this, ptensor);
  ptensor->set_tensor<T>(ff, dim_sizes, data);
  return true;
}

template <typename T>
bool TensorBase::get_tensor(FFModel const *ff, T *data, bool get_gradients) {
  ParallelTensor ptensor = nullptr;
  ff->get_parallel_tensor_from_tensor(this, ptensor);
  ptensor->get_tensor<T>(ff, data, get_gradients);
  return true;
}

template <typename T>
bool TensorBase::get_output_parallel_tensor(FFModel const *ff,
                                            T *data,
                                            bool get_gradients) {
  ParallelTensor parallel_tensor = nullptr;
  Op *final_operator = ff->get_final_operator();
  assert(final_operator->numOutputs == 1);
  parallel_tensor = final_operator->outputs[0];
  parallel_tensor->get_tensor<T>(ff, data, get_gradients);
  return true;
}

bool ParallelTensorShape::is_valid() const {
  bool used[MAX_TENSOR_DIM];
  std::fill_n(used, MAX_TENSOR_DIM, false);

  for (int i = 0; i < this->num_dims; i++) {
    ParallelDim const &dim = this->dims[i];
    assert(dim.size > 0);
    assert(dim.degree != ParallelDim::UNKNOWN_DEGREE);
    assert(dim.degree >= 1);
    assert(dim.parallel_idx != ParallelDim::UNKNOWN_INDEX);
    assert(dim.parallel_idx < MAX_TENSOR_DIM);
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
    assert(!used[i]);
  }

  return true;
}

bool ParallelTensorShape::operator==(ParallelTensorShape const &other) const {
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

bool ParallelTensorShape::operator!=(ParallelTensorShape const &other) const {
  return !(*this == other);
}

size_t ParallelTensorShape::get_piece_size() const {
  size_t piece_size = data_type_size(this->data_type);
  for (int i = 0; i < this->num_dims; i++) {
    piece_size *= this->dims[i].size / this->dims[i].degree;
  }
  return piece_size;
}

RecordFormatter ParallelTensorShape::as_dot() const {
  RecordFormatter r;
  for (int i = 0; i < this->num_dims; i++) {
    std::ostringstream oss;
    oss << "" << this->dims[i].size << "/" << this->dims[i].degree;
    r << oss.str();
  }
  return r;
}

std::unordered_map<int, int>
    ParallelTensorShape::get_mv_dim_to_tensor_dim_mapping() const {
  std::unordered_map<int, int> result;
  for (int i = 0; i < this->num_dims; i++) {
    int machine_view_dim = this->dims[i].parallel_idx;
    if (machine_view_dim != -1) {
      assert(result.find(machine_view_dim) == result.end());
      result[machine_view_dim] = i;
    }
  }
  return result;
}

std::unordered_map<int, int>
    ParallelTensorShape::get_tensor_dim_to_mv_dim_mapping() const {
  std::unordered_map<int, int> result;
  for (auto const &kv : this->get_mv_dim_to_tensor_dim_mapping()) {
    assert(result.find(kv.second) == result.end());
    result[kv.second] = kv.first;
  }
  return result;
}

bool ParallelTensorBase::update_parallel_ids(int numdim, ParallelDim *dims) {
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

ParallelTensorBase::ParallelTensorBase(ParallelTensorBase const &rhs) {
  parallel_tensor_guid = rhs.parallel_tensor_guid;
  num_dims = rhs.num_dims;
  for (int i = 0; i < num_dims; i++) {
    dims[i] = rhs.dims[i];
  }
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

void ParallelTensorBase::inline_map(FFConfig &config) {
  printf("inline map tensor\n");
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;

  RegionRequirement region_req(region, READ_WRITE, EXCLUSIVE, region);
  region_req.add_field(FID_DATA);
  InlineLauncher inline_launcher(region_req);
  physical_region = runtime->map_region(ctx, inline_launcher);
  physical_region.wait_until_valid();
}

void ParallelTensorBase::inline_unmap(FFConfig &config) {
  printf("inline unmap tensor\n");
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  assert(physical_region.is_valid() == true);
  runtime->unmap_region(ctx, physical_region);
}

template <typename T>
T *ParallelTensorBase::get_raw_ptr(FFConfig &config) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  RegionRequirement region_req(region, READ_WRITE, EXCLUSIVE, region);
  region_req.add_field(FID_DATA);
  T *raw_ptr = NULL;
  if (num_dims == 1) {
    TensorAccessorW<T, 1> acc(
        physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T *)acc.ptr;
  } else if (num_dims == 2) {
    TensorAccessorW<T, 2> acc(
        physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T *)acc.ptr;
  } else if (num_dims == 3) {
    TensorAccessorW<T, 3> acc(
        physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T *)acc.ptr;
  } else if (num_dims == 4) {
    TensorAccessorW<T, 4> acc(
        physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T *)acc.ptr;
  } else if (num_dims == 5) {
    TensorAccessorW<T, 5> acc(
        physical_region, region_req, FID_DATA, ctx, runtime, true);
    raw_ptr = (T *)acc.ptr;
  } else {
    printf("wrong num_dims %d", num_dims);
    assert(0);
  }
  return raw_ptr;
}

void ParallelTensorBase::attach_raw_ptr(FFConfig &config,
                                        void *raw_ptr,
                                        bool column_major) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  AttachLauncher launcher(EXTERNAL_INSTANCE, region, region);
  std::vector<FieldID> fields(1, FID_DATA);
  const Memory local_sysmem =
      Machine::MemoryQuery(Machine::get_machine())
          .has_affinity_to(runtime->get_executing_processor(ctx))
          .only_kind(Memory::SYSTEM_MEM)
          .first();
  launcher.attach_array_soa(raw_ptr, column_major, fields, local_sysmem);
  physical_region = runtime->attach_external_resource(ctx, launcher);
}

void ParallelTensorBase::detach_raw_ptr(FFConfig &config) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  runtime->detach_external_resource(ctx, physical_region);
}

template <typename T>
bool ParallelTensorBase::get_input_sub_tensor_via_mappings(
    ParallelConfig const &pc, ParallelTensorBase &tensor) const {
  if (pc.nDims != num_dims) {
    printf("Could not get input subtensor because the number of dimensions do "
           "not match: %d != %d\n",
           pc.nDims,
           num_dims);
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

bool ParallelTensorBase::get_sub_tensor(MachineView const &mv,
                                        ParallelTensorBase &sub_tensor) const {
  sub_tensor.num_dims = this->num_dims;
  for (int i = 0; i < sub_tensor.num_dims; i++) {
    sub_tensor.dims[i] = this->dims[i];
    if (this->dims[i].parallel_idx != -1) {
      int idx = this->dims[i].parallel_idx;
      assert(idx >= 0);
      assert(this->dims[i].degree == mv.dim[idx]);
      sub_tensor.dims[i].size /= mv.dim[idx];
      sub_tensor.dims[i].degree /= mv.dim[idx];
    }
  }
  return true;
}

bool ParallelTensorBase::get_input_sub_tensor(ParallelConfig const &pc,
                                              ParallelTensorBase &tensor,
                                              OperatorType type) {
  // TODO: consider reduction dim for conv2d and linear
  switch (type) {
    case OP_FLAT: {
      assert(pc.nDims == 3 &&
             "Invalid dimension for parallel config of OP_FLAT");

      tensor.num_dims = this->num_dims;
      for (int i = 0; i < 3; i++) {
        assert(tensor.dims[i].size % pc.dim[i] == 0);
        tensor.dims[i].size = tensor.dims[i].size / pc.dim[i];
      }
      break;
    }
    case OP_RESHAPE: {
      for (int i = 0; i < pc.nDims - 1; i++) {
        assert(pc.dim[i] == 1 && "Assuming data parallel for RESHAPE");
      }
      int batchDim = pc.dim[pc.nDims - 1];
      if (dims[num_dims - 1].size % batchDim != 0) {
        printf("Could not get input subtensor because the dimension is not "
               "divisiable: %d %% %d != 0\n",
               dims[num_dims - 1].size,
               batchDim);
      }
      tensor.num_dims = num_dims;
      for (int i = num_dims - 2; i >= 0; i--) {
        tensor.dims[i].size = dims[i].size;
      }
      tensor.dims[num_dims - 1].size = dims[num_dims - 1].size / batchDim;
      break;
    }
    case OP_LINEAR: {
      if (pc.nDims != num_dims) {
        printf("Could not get input subtensor because the number of dimensions "
               "do not match: %d != %d\n",
               pc.nDims,
               num_dims);
        return false;
      }
      tensor.num_dims = num_dims;
      for (int i = 0; i < num_dims; i++) {
        if (dims[i].size % pc.dim[i] != 0) {
          printf("Could not get input subtensor because the given dimension is "
                 "not divisible: %d %% %d != 0\n",
                 dims[i].size,
                 pc.dim[i]);
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
    default: {
      if (pc.nDims != num_dims) {
        printf("Could not get input subtensor because the number of dimensions "
               "do not match: %d != %d\n",
               pc.nDims,
               num_dims);
        return false;
      }
      for (int i = 0; i < num_dims; i++) {
        if (dims[i].size % pc.dim[i] != 0) {
          printf("Could not get input subtensor because the given dimension is "
                 "not divisible: %d %% %d != 0\n",
                 dims[i].size,
                 pc.dim[i]);
          return false;
        }
      }
      tensor.num_dims = num_dims;
      for (int i = 0; i < num_dims; i++) {
        tensor.dims[i].size = dims[i].size / pc.dim[i];
      }
      tensor.data_type = data_type;
    } break;
  }
  return true;
}

bool ParallelTensorBase::get_output_sub_tensor(ParallelConfig const &pc,
                                               ParallelTensorBase &tensor,
                                               OperatorType type) {
  if (pc.nDims != num_dims) {
    printf("Could not get output subtensor because the number of dimensions do "
           "not match: %d != %d\n",
           pc.nDims,
           num_dims);
    return false;
  }
  for (int i = 0; i < num_dims; i++) {
    if (dims[i].size % pc.dim[i] != 0) {
      printf("Could not get output subtensor because the given dimension is "
             "not divisible: %d %% %d != 0\n",
             dims[i].size,
             pc.dim[i]);
      return false;
    }
  }
  tensor.num_dims = num_dims;
  for (int i = 0; i < num_dims; i++) {
    tensor.dims[i].size = dims[i].size / pc.dim[i];
  }
  tensor.data_type = data_type;
  return true;
}

size_t ParallelTensorBase::get_owner_independent_hash() const {
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

size_t ParallelTensorBase::get_volume() const {
  size_t volume = 1;
  for (int i = 0; i < num_dims; i++) {
    volume *= dims[i].size;
  }
  return volume;
}

size_t ParallelTensorBase::get_total_num_parts() const {
  size_t parts = 1;
  for (int i = 0; i < num_dims; i++) {
    parts *= dims[i].degree;
  }
  return parts;
}

int ParallelTensorBase::get_num_replica_dims() const {
  return this->get_shape().get_num_replica_dims();
}

int ParallelTensorBase::get_num_replicas() const {
  return this->get_shape().get_num_replicas();
}

Domain ParallelTensorBase::get_domain() const {
  Domain d;
  d.dim = this->num_dims;
  for (int i = 0; i < this->num_dims; i++) {
    d.rect_data[i] = 0;
    d.rect_data[i + d.dim] = this->dims[i].size - 1;
  }
  return d;
}

bool ParallelTensorBase::check_valid() const {
  bool used[MAX_TENSOR_DIM];
  for (int i = 0; i < MAX_TENSOR_DIM; i++) {
    used[i] = false;
  }
  for (int i = 0; i < num_dims; i++) {
    if (dims[i].size < 0) {
      return false;
    }
    if (dims[i].size % dims[i].degree != 0) {
      return false;
    }
    if (dims[i].parallel_idx > MAX_TENSOR_DIM) {
      return false;
    }
    assert(dims[i].parallel_idx >= -1);
    assert(dims[i].degree >= 1);
    if (dims[i].parallel_idx >= 0) {
      if (used[dims[i].parallel_idx]) {
        return false;
      }
      used[dims[i].parallel_idx] = true;
    }
  }
  assert(this->data_type != DT_NONE);
  int idx = 0;
  while (used[idx]) {
    idx++;
  }
  for (int i = idx; i < MAX_TENSOR_DIM; i++) {
    if (used[i]) {
      return false;
    }
  }
  return true;
}

void TensorBase::print(std::string const &name) const {
  printf("%s: sizes[", name.c_str());

  for (int i = 0; i < num_dims; i++) {
    printf("%d ", dims[i]);
  }
  printf("]\n");
}

void ParallelTensorBase::print(std::string const &name) const {
  printf("%s: sizes[", name.c_str());

  for (int i = 0; i < num_dims; i++) {
    printf("%d ", dims[i].size);
  }
  printf("] degree[");
  for (int i = 0; i < num_dims; i++) {
    printf("%d ", dims[i].degree);
  }
  printf("] parallel_ids[");
  for (int i = 0; i < num_dims; i++) {
    printf("%d ", dims[i].parallel_idx);
  }
  printf("]\n");
}

ParallelTensorShape::ParallelTensorShape(int num_dims,
                                         ParallelDim const dims[MAX_TENSOR_DIM],
                                         DataType data_type)
    : num_dims(num_dims), data_type(data_type) {
  for (int i = 0; i < num_dims; i++) {
    this->dims[i] = dims[i];
  }
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

int ParallelTensorShape::get_num_replica_dims() const {
  int num_replica_dims = 0;
  for (int i = 0; i < this->num_dims; i++) {
    if (this->dims[i].is_replica_dim) {
      num_replica_dims++;
    }
  }

  return num_replica_dims;
}

int ParallelTensorShape::get_num_replicas() const {
  int num_replicas = 1;
  for (int i = 0; i < this->num_dims; i++) {
    if (this->dims[i].is_replica_dim) {
      num_replicas *= this->dims[i].degree;
    }
  }

  return num_replicas;
}

std::ostream &operator<<(std::ostream &s, ParallelTensorShape const &shape) {
  s << "[ ";
  for (int i = 0; i < shape.num_dims; i++) {
    s << shape.dims[i].size << "/" << shape.dims[i].degree << " ";
  }
  s << "]";

  return s;
}

}; // namespace FlexFlow

namespace std {
size_t hash<FlexFlow::ParallelTensorShape>::operator()(
    FlexFlow::ParallelTensorShape const &shape) const {
  size_t key = 0;
  hash_combine(key, shape.num_dims);
  for (int i = 0; i < shape.num_dims; i++) {
    hash_combine(key, shape.dims[i].size);
    hash_combine(key, shape.dims[i].degree);
  }
  return key;
}
}; // namespace std

namespace FlexFlow {

bool ParallelTensorBase::is_valid_machine_view(MachineView const &view) const {
  int is_dim = 0;
  for (int i = 0; i < num_dims; i++) {
    if (dims[i].parallel_idx != -1) {
      is_dim++;
      if (dims[i].parallel_idx > view.ndims) {
        return false;
      }
      if (view.dim[dims[i].parallel_idx] != dims[i].degree) {
        return false;
      }
    }
  }
  if (is_dim == 0) {
    is_dim = 1;
  }
  if (is_dim != view.ndims) {
    return false;
  }
  if (get_total_num_parts() != view.num_parts()) {
    return false;
  }
  return true;
}

template <typename T>
bool ParallelTensorBase::set_tensor(FFModel const *ff,
                                    std::vector<int> const &dim_sizes,
                                    T const *data) {
  Context ctx = ff->config.lg_ctx;
  Runtime *runtime = ff->config.lg_hlr;
  // TODO: check data type matches
  // TODO: Currently we use a task launch, change to index launch for NCCL
  // parameter
  size_t volume = 1, num_replicas = 1;
  if (sync_type == ParameterSyncType::NCCL) {
    // Domain domain = runtime->get_index_space_domain(ctx, parallel_is);
    // num_replicas = domain.get_volume();
    for (int i = 0; i < this->num_dims; i++) {
      if (this->dims[i].is_replica_dim) {
        num_replicas *= this->dims[i].size;
      }
    }
  } else if (sync_type == ParameterSyncType::PS) {
    num_replicas = 1;
  } else {
    num_replicas = 1;
  }
  for (size_t i = 0; i < dim_sizes.size(); i++) {
    volume = volume * dim_sizes[i];
  }
  RegionRequirement req(region, WRITE_ONLY, EXCLUSIVE, region);
  req.add_field(FID_DATA);
  InlineLauncher launcher(req);
  PhysicalRegion pr = runtime->map_region(ctx, launcher);
  pr.wait_until_valid();
  switch (num_dims) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorW<T, DIM> acc(pr, req, FID_DATA, ctx, runtime, false);       \
    assert(acc.rect.volume() == volume * num_replicas);                        \
    T *ptr = acc.ptr;                                                          \
    for (size_t i = 0; i < num_replicas; i++) {                                \
      memcpy(ptr, data, volume * sizeof(T));                                   \
      ptr += volume;                                                           \
    }                                                                          \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      // Unsupported dim
      assert(false);
  }
  runtime->unmap_region(ctx, pr);
  return true;
}

template <typename T>
bool ParallelTensorBase::get_tensor(FFModel const *ff,
                                    T *data,
                                    bool get_gradients) {
  Context ctx = ff->config.lg_ctx;
  Runtime *runtime = ff->config.lg_hlr;
  LogicalRegion weight_lr = LogicalRegion::NO_REGION;
  if (sync_type == ParameterSyncType::PS) {
    weight_lr = get_gradients ? region_grad : region;
  } else {
    assert(owner_op != NULL);
    Domain domain = runtime->get_index_space_domain(ctx, parallel_is);
    switch (domain.get_dim()) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    DomainPoint point = Point<DIM>::ZEROES();                                  \
    weight_lr = runtime->get_logical_subregion_by_color(                       \
        ctx, get_gradients ? part_grad : part, point);                         \
    break;                                                                     \
  }
      LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    }
  }
  // TODO: check data type matches
  size_t volume = 1;
  for (int i = 0; i < num_dims; i++) {
    volume = volume * dims[i].size / dims[i].degree;
  }
  RegionRequirement req(
      weight_lr, READ_ONLY, EXCLUSIVE, get_gradients ? region_grad : region);
  req.add_field(FID_DATA);
  InlineLauncher launcher(req);
  PhysicalRegion pr = runtime->map_region(ctx, launcher);
  pr.wait_until_valid();
  switch (num_dims) {
#define DIMFUNC(DIM)                                                           \
  case DIM: {                                                                  \
    TensorAccessorR<T, DIM> acc(pr, req, FID_DATA, ctx, runtime);              \
    assert(acc.rect.volume() == volume);                                       \
    memcpy(data, acc.ptr, volume * sizeof(T));                                 \
    break;                                                                     \
  }
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      // Unsupported dim
      assert(false);
  }
  runtime->unmap_region(ctx, pr);
  return true;
}

template <typename T>
bool ParallelTensorBase::tensor_equal(FFConfig &config,
                                      ParallelTensorBase &tensor) {
  Context ctx = config.lg_ctx;
  Runtime *runtime = config.lg_hlr;
  TaskLauncher launcher(TENSOR_EQUAL_TASK_ID,
                        TaskArgument(&num_dims, sizeof(num_dims)));
  launcher.add_region_requirement(
      RegionRequirement(region, READ_ONLY, EXCLUSIVE, region));
  launcher.add_field(0, FID_DATA);
  launcher.add_region_requirement(
      RegionRequirement(tensor.region, READ_ONLY, EXCLUSIVE, tensor.region));
  launcher.add_field(1, FID_DATA);
  Future result = runtime->execute_task(ctx, launcher);
  bool equals = result.get_result<bool>();
  return equals;
}

bool ParallelTensorBase::tensor_equal_task(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  assert(regions.size() == 2);
  int dim = *(int const *)task->args;
  switch (dim) {
#define DIMFUNC(DIM)                                                           \
  case DIM:                                                                    \
    return tensor_equal_task_with_dim<DIM>(task, regions, ctx, runtime);
    LEGION_FOREACH_N(DIMFUNC)
#undef DIMFUNC
    default:
      assert(false);
  }
  assert(false);
}

template <int NDIM>
bool ParallelTensorBase::tensor_equal_task_with_dim(
    Task const *task,
    std::vector<PhysicalRegion> const &regions,
    Context ctx,
    Runtime *runtime) {
  TensorAccessorR<float, NDIM> acc1(
      regions[0], task->regions[0], FID_DATA, ctx, runtime);
  TensorAccessorR<float, NDIM> acc2(
      regions[1], task->regions[1], FID_DATA, ctx, runtime);
  float const *data1 = acc1.ptr;
  float const *data2 = acc2.ptr;
  bool equal = true;
  for (int i = 0; i < acc1.rect.volume(); i++) {
    if (data1[i] != data2[i]) {
      equal = false;
      break;
    }
  }
  return equal;
}

template float *ParallelTensorBase::get_raw_ptr<float>(FFConfig &config);
template int32_t *ParallelTensorBase::get_raw_ptr<int32_t>(FFConfig &config);

template bool TensorBase::set_tensor<float>(FFModel const *ff,
                                            std::vector<int> const &dims,
                                            float const *data);
template bool TensorBase::get_tensor<float>(FFModel const *ff,
                                            float *data,
                                            bool get_gradients);
template bool TensorBase::set_tensor<double>(FFModel const *ff,
                                             std::vector<int> const &dims,
                                             double const *data);
template bool TensorBase::get_tensor<double>(FFModel const *ff,
                                             double *data,
                                             bool get_gradients);
template bool TensorBase::set_tensor<int32_t>(FFModel const *ff,
                                              std::vector<int> const &dims,
                                              int32_t const *data);
template bool TensorBase::get_tensor<int32_t>(FFModel const *ff,
                                              int32_t *data,
                                              bool get_gradients);
template bool TensorBase::set_tensor<int64_t>(FFModel const *ff,
                                              std::vector<int> const &dims,
                                              int64_t const *data);
template bool TensorBase::get_tensor<int64_t>(FFModel const *ff,
                                              int64_t *data,
                                              bool get_gradients);

template bool ParallelTensorBase::set_tensor<half>(FFModel const *ff,
                                                   std::vector<int> const &dims,
                                                   half const *data);
template bool ParallelTensorBase::get_tensor<half>(FFModel const *ff,
                                                   half *data,
                                                   bool get_gradients);

template bool ParallelTensorBase::set_tensor<char>(FFModel const *ff,
                                                   std::vector<int> const &dims,
                                                   char const *data);
template bool ParallelTensorBase::get_tensor<char>(FFModel const *ff,
                                                   char *data,
                                                   bool get_gradients);

template bool ParallelTensorBase::set_tensor<float>(
    FFModel const *ff, std::vector<int> const &dims, float const *data);
template bool ParallelTensorBase::get_tensor<float>(FFModel const *ff,
                                                    float *data,
                                                    bool get_gradients);
template bool ParallelTensorBase::set_tensor<double>(
    FFModel const *ff, std::vector<int> const &dims, double const *data);
template bool ParallelTensorBase::get_tensor<double>(FFModel const *ff,
                                                     double *data,
                                                     bool get_gradients);
template bool ParallelTensorBase::set_tensor<int32_t>(
    FFModel const *ff, std::vector<int> const &dims, int32_t const *data);
template bool ParallelTensorBase::get_tensor<int32_t>(FFModel const *ff,
                                                      int32_t *data,
                                                      bool get_gradients);
template bool ParallelTensorBase::set_tensor<int64_t>(
    FFModel const *ff, std::vector<int> const &dims, int64_t const *data);
template bool ParallelTensorBase::get_tensor<int64_t>(FFModel const *ff,
                                                      int64_t *data,
                                                      bool get_gradients);

template bool
    ParallelTensorBase::tensor_equal<float>(FFConfig &config,
                                            ParallelTensorBase &tensor);

template bool TensorBase::get_output_parallel_tensor<float>(FFModel const *ff,
                                                            float *data,
                                                            bool get_gradients);

}; // namespace FlexFlow
