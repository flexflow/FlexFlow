include(FetchContent)

FetchContent_Declare(json URL https://github.com/nlohmann/json/releases/download/v3.10.5/json.tar.xz)
FetchContent_MakeAvailable(json)

list(APPEND FLEXFLOW_EXT_LIBRARIES nlohmann_json::nlohmann_json)
