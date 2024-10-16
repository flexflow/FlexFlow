#include "kernels/accessor.h"
#include "kernels/allocation.h"
#include "kernels/datatype_dispatch.h"

namespace FlexFlow {

void transfer_data_between_accessors(
    GenericTensorAccessorW &dst_accessor,
    GenericTensorAccessorR const &src_accessor) {
  size_t num_bytes = dst_accessor.shape.get_volume() *
                     size_of_datatype(dst_accessor.data_type);

  DeviceType dst_device_type = dst_accessor.device_type;
  DeviceType src_device_type = src_accessor.device_type;

  if (src_device_type == DeviceType::CPU &&
      dst_device_type == DeviceType::CPU) {
    memcpy(dst_accessor.ptr, src_accessor.ptr, num_bytes);
  } else if (src_device_type == DeviceType::CPU &&
             dst_device_type == DeviceType::GPU) {
    checkCUDA(cudaMemcpy(
        dst_accessor.ptr, src_accessor.ptr, num_bytes, cudaMemcpyHostToDevice));
  } else if (src_device_type == DeviceType::GPU &&
             dst_device_type == DeviceType::CPU) {
    checkCUDA(cudaMemcpy(
        dst_accessor.ptr, src_accessor.ptr, num_bytes, cudaMemcpyDeviceToHost));
  } else {
    checkCUDA(cudaMemcpy(dst_accessor.ptr,
                         src_accessor.ptr,
                         num_bytes,
                         cudaMemcpyDeviceToDevice));
  }
}

void transfer_data_between_accessors(
    GenericTensorAccessorW &dst_accessor,
    GenericTensorAccessorW const &src_accessor) {
  GenericTensorAccessorR r_src_accessor =
      read_only_accessor_from_write_accessor(src_accessor);
  transfer_data_between_accessors(dst_accessor, r_src_accessor);
}

GenericTensorAccessorW::GenericTensorAccessorW(
    DataType data_type,
    ArrayShape const &shape,
    void *ptr,
    DeviceType device_type = DeviceType::GPU)
    : data_type(data_type), shape(shape), ptr(ptr), device_type(device_type) {}

std::tuple<DataType const &,
           ArrayShape const &,
           void *const &,
           DeviceType const &>
    GenericTensorAccessorW::tie() const {
  return std::tie(this->data_type, this->shape, this->ptr, this->device_type);
}

size_t GenericTensorAccessorW::calculate_index_offset(
    std::initializer_list<size_t> const &indices) const {

  if (indices.size() != this->shape.num_dims()) {
    throw mk_runtime_error(fmt::format(
        "Number of indices ({}) does not match the number of dimensions ({}).",
        indices.size(),
        this->shape.num_dims()));
  }

  size_t offset = 0;
  size_t multiplier = 1;
  size_t cur_idx;
  auto it = indices.begin();

  for (size_t i = 0; i < this->shape.num_dims(); i++) {
    cur_idx = *it++;

    if (cur_idx >= this->shape.at(legion_dim_t(i))) {
      throw mk_runtime_error(
          fmt::format("In {} dimension, attempting to access index {} "
                      "when only {} indexes exist",
                      i,
                      cur_idx,
                      this->shape.at(legion_dim_t(i))));
    }

    offset += cur_idx * multiplier;
    multiplier *= this->shape.at(legion_dim_t(i));
  }

  return offset;
}

bool GenericTensorAccessorW::operator==(
    GenericTensorAccessorW const &other) const {
  return this->tie() == other.tie();
}

bool GenericTensorAccessorW::operator!=(
    GenericTensorAccessorW const &other) const {
  return this->tie() != other.tie();
}

int32_t *GenericTensorAccessorW::get_int32_ptr() const {
  return this->get<DataType::INT32>();
}

int64_t *GenericTensorAccessorW::get_int64_ptr() const {
  return this->get<DataType::INT64>();
}

float *GenericTensorAccessorW::get_float_ptr() const {
  return this->get<DataType::FLOAT>();
}

double *GenericTensorAccessorW::get_double_ptr() const {
  return this->get<DataType::DOUBLE>();
}

half *GenericTensorAccessorW::get_half_ptr() const {
  return this->get<DataType::HALF>();
}

std::string format_as(GenericTensorAccessorW const &a) {
  return fmt::format("<GenericTensorAccessorW data_type={} shape={} ptr={}>",
                     a.data_type,
                     a.shape,
                     a.ptr);
}

std::ostream &operator<<(std::ostream &s, GenericTensorAccessorW const &a) {
  return (s << fmt::to_string(a));
}

GenericTensorAccessorR::GenericTensorAccessorR(
    DataType data_type,
    ArrayShape const &shape,
    void const *ptr,
    DeviceType device_type = DeviceType::GPU)
    : data_type(data_type), shape(shape), ptr(ptr), device_type(device_type) {}

std::tuple<DataType const &,
           ArrayShape const &,
           void const *const &,
           DeviceType const &>
    GenericTensorAccessorR::tie() const {
  return std::tie(this->data_type, this->shape, this->ptr, this->device_type);
}

size_t GenericTensorAccessorR::calculate_index_offset(
    std::initializer_list<size_t> const &indices) const {

  if (indices.size() != this->shape.num_dims()) {
    throw mk_runtime_error(fmt::format(
        "Number of indices ({}) does not match the number of dimensions ({}).",
        indices.size(),
        this->shape.num_dims()));
  }

  size_t offset = 0;
  size_t multiplier = 1;
  size_t cur_idx;
  auto it = indices.begin();

  for (size_t i = 0; i < this->shape.num_dims(); i++) {
    cur_idx = *it++;

    if (cur_idx >= this->shape.at(legion_dim_t(i))) {
      throw mk_runtime_error(
          fmt::format("In {} dimension, attempting to access index {} "
                      "when only {} indexes exist",
                      i,
                      cur_idx,
                      this->shape.at(legion_dim_t(i))));
    }

    offset += cur_idx * multiplier;
    multiplier *= this->shape.at(legion_dim_t(i));
  }

  return offset;
}

bool GenericTensorAccessorR::operator==(
    GenericTensorAccessorR const &other) const {
  return this->tie() == other.tie();
}

bool GenericTensorAccessorR::operator!=(
    GenericTensorAccessorR const &other) const {
  return this->tie() != other.tie();
}

int32_t const *GenericTensorAccessorR::get_int32_ptr() const {
  return this->get<DataType::INT32>();
}

int64_t const *GenericTensorAccessorR::get_int64_ptr() const {
  return this->get<DataType::INT64>();
}

float const *GenericTensorAccessorR::get_float_ptr() const {
  return this->get<DataType::FLOAT>();
}

double const *GenericTensorAccessorR::get_double_ptr() const {
  return this->get<DataType::DOUBLE>();
}

half const *GenericTensorAccessorR::get_half_ptr() const {
  return get<DataType::HALF>();
}

std::string format_as(GenericTensorAccessorR const &a) {
  return fmt::format("<GenericTensorAccessorR data_type={} shape={} ptr={}>",
                     a.data_type,
                     a.shape,
                     a.ptr);
}

std::ostream &operator<<(std::ostream &s, GenericTensorAccessorR const &a) {
  return (s << fmt::to_string(a));
}

int32_t *get_int32_ptr(GenericTensorAccessorW const &a) {
  return get<DataType::INT32>(a);
}

int64_t *get_int64_ptr(GenericTensorAccessorW const &a) {
  return get<DataType::INT64>(a);
}

float *get_float_ptr(GenericTensorAccessorW const &a) {
  return get<DataType::FLOAT>(a);
}

double *get_double_ptr(GenericTensorAccessorW const &a) {
  return get<DataType::DOUBLE>(a);
}

half *get_half_ptr(GenericTensorAccessorW const &a) {
  return get<DataType::HALF>(a);
}

std::vector<int32_t *>
    get_int32_ptrs(std::vector<GenericTensorAccessorW> const &a) {
  return get<DataType::INT32>(a);
}

std::vector<int64_t *>
    get_int64_ptrs(std::vector<GenericTensorAccessorW> const &a) {
  return get<DataType::INT64>(a);
}

std::vector<float *>
    get_float_ptrs(std::vector<GenericTensorAccessorW> const &a) {
  return get<DataType::FLOAT>(a);
}

std::vector<double *>
    get_double_ptrs(std::vector<GenericTensorAccessorW> const &a) {
  return get<DataType::DOUBLE>(a);
}

std::vector<half *>
    get_half_ptrs(std::vector<GenericTensorAccessorW> const &a) {
  return get<DataType::HALF>(a);
}

int32_t const *get_int32_ptr(GenericTensorAccessorR const &a) {
  return get<DataType::INT32>(a);
}

int64_t const *get_int64_ptr(GenericTensorAccessorR const &a) {
  return get<DataType::INT64>(a);
}

float const *get_float_ptr(GenericTensorAccessorR const &a) {
  return get<DataType::FLOAT>(a);
}

double const *get_double_ptr(GenericTensorAccessorR const &a) {
  return get<DataType::DOUBLE>(a);
}

half const *get_half_ptr(GenericTensorAccessorR const &a) {
  return get<DataType::HALF>(a);
}

std::vector<int32_t const *>
    get_int32_ptrs(std::vector<GenericTensorAccessorR> const &a) {
  return get<DataType::INT32>(a);
}

std::vector<int64_t const *>
    get_int64_ptrs(std::vector<GenericTensorAccessorR> const &a) {
  return get<DataType::INT64>(a);
}

std::vector<float const *>
    get_float_ptrs(std::vector<GenericTensorAccessorR> const &a) {
  return get<DataType::FLOAT>(a);
}

std::vector<double const *>
    get_double_ptrs(std::vector<GenericTensorAccessorR> const &a) {
  return get<DataType::DOUBLE>(a);
}

std::vector<half const *>
    get_half_ptrs(std::vector<GenericTensorAccessorR> const &a) {
  return get<DataType::HALF>(a);
}

GenericTensorAccessorR read_only_accessor_from_write_accessor(
    GenericTensorAccessorW const &writable) {
  return GenericTensorAccessorR{writable.data_type,
                                writable.shape,
                                req<void const *>(writable.ptr),
                                writable.device_type};
}

bool is_shape_and_dtype_equal(GenericTensorAccessorW const &acc1,
                              GenericTensorAccessorW const &acc2) {
  return acc1.shape == acc2.shape && acc1.data_type == acc2.data_type;
}

bool shape_and_dtype_matches(GenericTensorAccessorW const &accessor,
                             ArrayShape const &expected_shape,
                             DataType const &expected_dtype) {
  return accessor.shape == expected_shape &&
         accessor.data_type == expected_dtype;
}

bool shape_and_dtype_matches(GenericTensorAccessorR const &accessor,
                             ArrayShape const &expected_shape,
                             DataType const &expected_dtype) {
  return accessor.shape == expected_shape &&
         accessor.data_type == expected_dtype;
}

std::pair<ArrayShape, DataType>
    get_shape_and_datatype(GenericTensorAccessorR const &accessor) {
  return std::make_pair(accessor.shape, accessor.data_type);
}

std::pair<ArrayShape, DataType>
    get_shape_and_datatype(GenericTensorAccessorW const &accessor) {
  return std::make_pair(accessor.shape, accessor.data_type);
}

template <DataType DT>
struct CopyTensorAccessorW {
  GenericTensorAccessorW operator()(GenericTensorAccessorW const &src_accessor,
                                    Allocator &allocator) {
    TensorShape shape =
        get_tensor_shape(src_accessor.shape, src_accessor.data_type);
    GenericTensorAccessorW dst_accessor = allocator.allocate_tensor(shape);

    transfer_data_between_accessors(dst_accessor, src_accessor);

    return dst_accessor;
  }
};

GenericTensorAccessorW
    copy_tensor_accessor_w(GenericTensorAccessorW const &src_accessor,
                           Allocator &allocator) {
  return DataTypeDispatch1<CopyTensorAccessorW>{}(
      src_accessor.data_type, src_accessor, std::ref(allocator));
}

template <DataType DT>
struct CopyTensorAccessorR {
  GenericTensorAccessorR operator()(GenericTensorAccessorR const &src_accessor,
                                    Allocator &allocator) {
    TensorShape shape =
        get_tensor_shape(src_accessor.shape, src_accessor.data_type);
    GenericTensorAccessorW dst_accessor = allocator.allocate_tensor(shape);

    transfer_data_between_accessors(dst_accessor, src_accessor);

    return read_only_accessor_from_write_accessor(dst_accessor);
  }
};

GenericTensorAccessorR
    copy_tensor_accessor_r(GenericTensorAccessorR const &src_accessor,
                           Allocator &allocator) {
  return DataTypeDispatch1<CopyTensorAccessorR>{}(
      src_accessor.data_type, src_accessor, std::ref(allocator));
}

} // namespace FlexFlow
