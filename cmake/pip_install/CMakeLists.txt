# Use setup.py script to re-install the Python bindings library with the right library paths
if (FF_USE_PYTHON)
    execute_process(COMMAND ${Python_EXECUTABLE} -c "import site, os; print([pkg for func in (site.getsitepackages(), site.getusersitepackages()) for pkg in ([func] if isinstance(func, str) else func) if os.access(pkg, os.W_OK)][0])" OUTPUT_VARIABLE PY_DEST OUTPUT_STRIP_TRAILING_WHITESPACE)  
    if(FF_BUILD_FROM_PYPI)
        install(CODE "execute_process(COMMAND ${CMAKE_COMMAND} -E echo \"Editing path to Legion library using path: ${PY_DEST}/flexflow/lib \")")
        # CMAKE_CURRENT_SOURCE_DIR=/usr/FlexFlow/cmake/pip_install 
        # Legion_BINARY_DIR=/usr/FlexFlow/build/<something>/deps/legion
        install(CODE "execute_process(COMMAND ${Python_EXECUTABLE} ${CMAKE_CURRENT_SOURCE_DIR}/../../deps/legion/bindings/python/setup.py install --cmake-build-dir ${Legion_BINARY_DIR}/runtime --prefix ${PY_DEST}/flexflow ${Legion_PYTHON_EXTRA_INSTALL_ARGS} WORKING_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/../../deps/legion/bindings/python)")
    endif()
endif()
