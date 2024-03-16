include(aliasing)

if (FF_USE_EXTERNAL_JSON)
  find_package(nlohmann_json REQUIRED)

  alias_library(json nlohmann_json)
else()
  set(JSON_BuildTests OFF CACHE INTERNAL "")
  add_subdirectory(${CMAKE_CURRENT_SOURCE_DIR}/deps/json)

  alias_library(json nlohmann_json::nlohmann_json)
endif()
