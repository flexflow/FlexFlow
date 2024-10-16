#include "doctest/doctest.h"
#include "kernels/managed_per_device_ff_handle.h"

using namespace ::FlexFlow;

TEST_SUITE(FF_TEST_SUITE) {
  TEST_CASE("Test Managed Per Device FF Handle") {
    ManagedPerDeviceFFHandle base_handle{};

    SUBCASE("Test ManagedPerDeviceFFHandle Constructor") {
      CHECK(base_handle.raw_handle().workSpaceSize == 1024 * 1024);
      CHECK(base_handle.raw_handle().allowTensorOpMathConversion == true);
    }

    SUBCASE("Test ManagedPerDeviceFFHandle Move Constructor") {
      PerDeviceFFHandle const *base_handle_ptr = &base_handle.raw_handle();

      ManagedPerDeviceFFHandle new_handle(std::move(base_handle));

      CHECK(&base_handle.raw_handle() == nullptr);
      CHECK(&new_handle.raw_handle() == base_handle_ptr);
    }

    SUBCASE("Test ManagedPerDeviceFFHandle Assignment Operator") {
      PerDeviceFFHandle const *base_handle_ptr = &base_handle.raw_handle();

      ManagedPerDeviceFFHandle new_handle{};
      new_handle = std::move(base_handle);

      CHECK(&base_handle.raw_handle() == nullptr);
      CHECK(&new_handle.raw_handle() == base_handle_ptr);
    }
  }
}
