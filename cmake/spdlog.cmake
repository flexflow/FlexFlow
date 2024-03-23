include(aliasing)

if (FF_USE_EXTERNAL_SPDLOG)
  find_package(spdlog REQUIRED)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/spdlog)
endif()

add_library(spdlog INTERFACE)
target_link_libraries(spdlog INTERFACE spdlog::spdlog)
target_compile_definitions(spdlog INTERFACE SPDLOG_FMT_EXTERNAL)
