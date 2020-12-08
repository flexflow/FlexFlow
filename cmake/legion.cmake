include_directories(${LEGION_ROOT}/include ${LEGION_ROOT}/include/mappers)
#set(CMAKE_FIND_DEBUG_MODE TRUE)
find_library(Legion_LIBRARIES
        NAMES legion
	PATHS ${LEGION_ROOT}/lib64 ${LEGION_ROOT}/lib
	NO_DEFAULT_PATH)
find_library(Realm_LIBRARIES
        NAMES realm
	PATHS ${LEGION_ROOT}/lib64 ${LEGION_ROOT}/lib 
	NO_DEFAULT_PATH)
message( STATUS "Legion root : ${LEGION_ROOT}" )
message( STATUS "Legion libraries : ${Legion_LIBRARIES}" )
message( STATUS "Realm libraries : ${Realm_LIBRARIES}" )
#list(APPEND FLEXFLOW_EXT_LIBRARIES ${Legion_LIBRARIES} ${Realm_LIBRARIES})
#target_link_libraries(flexflow ${LEGION_LIB} ${REALM_LIB})
#add_library(legion STATIC IMPORTED)
#set_property(TARGET legion PROPERTY IMPORTED_LOCATION ${LEGION_ROOT}/lib/liblegion.a)
#add_library(realm STATIC IMPORTED)
#set_property(TARGET realm PROPERTY IMPORTED_LOCATION ${LEGION_ROOT}/lib/librealm.a)
