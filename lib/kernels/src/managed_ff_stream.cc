#include "kernels/managed_ff_stream.h"

namespace FlexFlow {

ManagedFFStream::ManagedFFStream() : stream(new ffStream_t) {
  checkCUDA(cudaStreamCreate(this->stream));
}

ManagedFFStream::ManagedFFStream(ManagedFFStream &&other) noexcept
    : stream(std::exchange(other.stream, nullptr)) {}

ManagedFFStream &ManagedFFStream::operator=(ManagedFFStream &&other) noexcept {
  if (this != &other) {
    this->cleanup();
    this->stream = std::exchange(other.stream, nullptr);
  }
  return *this;
}

ManagedFFStream::~ManagedFFStream() {
  this->cleanup();
}

void ManagedFFStream::cleanup() {
  if (this->stream != nullptr) {
    checkCUDA(cudaStreamDestroy(*this->stream));
    delete this->stream;
  }
}

ffStream_t const &ManagedFFStream::raw_stream() const {
  return *this->stream;
}

} // namespace FlexFlow
