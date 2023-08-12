# Use setup.py script to re-install the Python bindings library with the right library paths
if (FF_USE_PYTHON)
    execute_process(COMMAND ${PYTHON_EXECUTABLE} -c "from distutils import sysconfig; print(sysconfig.get_python_lib(plat_specific=False,standard_lib=False))" OUTPUT_VARIABLE PY_DEST OUTPUT_STRIP_TRAILING_WHITESPACE)  
    if(FF_BUILD_FROM_PYPI)
        install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E echo \"Editing path to Legion library using path: ${PY_DEST}/flexflow/lib \")")
        # CMAKE_CURRENT_SOURCE_DIR=/usr/FlexFlow/cmake/pip_install 
        # Legion_BINARY_DIR=/usr/FlexFlow/build/<something>/deps/legion
        install(CODE "execute_process(COMMAND ${PYTHON_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../deps/legion/bindings/python/setup.py install --cmake-build-dir ${Legion_BINARY_DIR}/runtime --prefix ${PY_DEST}/flexflow ${Legion_PYTHON_EXTRA_INSTALL_ARGS} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../deps/legion/bindings/python)")
    endif()
endif()
