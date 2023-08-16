#----------------------------------------------------------------
# Generated CMake target import file for configuration "Debug".
#----------------------------------------------------------------

# Commands may need to know the format version.
set(CMAKE_IMPORT_FILE_VERSION 1)

# Import target "Legion::RealmRuntime" for configuration "Debug"
set_property(TARGET Legion::RealmRuntime APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(Legion::RealmRuntime PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/librealm.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS Legion::RealmRuntime )
list(APPEND _IMPORT_CHECK_FILES_FOR_Legion::RealmRuntime "${_IMPORT_PREFIX}/lib/librealm.a" )

# Import target "Legion::LegionRuntime" for configuration "Debug"
set_property(TARGET Legion::LegionRuntime APPEND PROPERTY IMPORTED_CONFIGURATIONS DEBUG)
set_target_properties(Legion::LegionRuntime PROPERTIES
  IMPORTED_LINK_INTERFACE_LANGUAGES_DEBUG "CUDA;CXX"
  IMPORTED_LOCATION_DEBUG "${_IMPORT_PREFIX}/lib/liblegion.a"
  )

list(APPEND _IMPORT_CHECK_TARGETS Legion::LegionRuntime )
list(APPEND _IMPORT_CHECK_FILES_FOR_Legion::LegionRuntime "${_IMPORT_PREFIX}/lib/liblegion.a" )

# Commands beyond this point should not need to know the version.
set(CMAKE_IMPORT_FILE_VERSION)
