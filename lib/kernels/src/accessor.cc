#include "kernels/accessor.h"

namespace FlexFlow {

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
  return GenericTensorAccessorR{
      writable.data_type, writable.shape, req<void const *>(writable.ptr)};
}

} // namespace FlexFlow
