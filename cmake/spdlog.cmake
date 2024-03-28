include(aliasing)

if (FF_USE_EXTERNAL_SPDLOG)
  find_package(spdlog REQUIRED)
else()
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/spdlog)
endif()

add_library(ff_spdlog INTERFACE)
target_link_libraries(ff_spdlog INTERFACE spdlog::spdlog)
target_compile_definitions(ff_spdlog INTERFACE SPDLOG_FMT_EXTERNAL)
alias_library(spdlog ff_spdlog)
