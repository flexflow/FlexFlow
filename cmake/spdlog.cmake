include(aliasing)

if (FF_USE_EXTERNAL_SPDLOG)
  find_package(spdlog REQUIRED)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/spdlog)

  alias_library(spdlog spdlog::spdlog)
endif()
