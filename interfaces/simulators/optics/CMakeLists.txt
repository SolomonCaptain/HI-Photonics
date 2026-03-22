cmake_minimum_required(VERSION 3.15)
project(optics VERSION 0.1.0 LANGUAGES CXX)

# 设置 C++ 标准
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# Windows 特定设置
if(WIN32)
    # 使用静态运行时库避免依赖问题
    set(CMAKE_MSVC_RUNTIME_LIBRARY "MultiThreaded$<$<CONFIG:Debug>:Debug>")
    # 添加定义
    add_definitions(-DWIN32_LEAN_AND_MEAN -DNOMINMAX)
    # 禁用某些 MSVC 警告
    add_compile_options(/wd4251 /wd4275 /wd4244)
endif()

# 查找 Python 和 pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)

# 尝试查找 pybind11
find_package(pybind11 CONFIG)

# 如果找不到 pybind11，尝试使用 Python 的 pybind11
if(NOT pybind11_FOUND)
    execute_process(
        COMMAND "${Python3_EXECUTABLE}" -c "import pybind11; print(pybind11.get_cmake_dir())"
        OUTPUT_VARIABLE pybind11_DIR
        OUTPUT_STRIP_TRAILING_WHITESPACE
        RESULT_VARIABLE pybind11_result
    )
    if(pybind11_result EQUAL 0 AND EXISTS "${pybind11_DIR}")
        list(APPEND CMAKE_PREFIX_PATH "${pybind11_DIR}")
        find_package(pybind11 CONFIG REQUIRED)
    else()
        message(FATAL_ERROR "pybind11 not found. Please install it: pip install pybind11")
    endif()
endif()

# 包含目录
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# 源文件
set(SOURCES
    src/grid.cpp
    src/pml.cpp
    src/source.cpp
    src/monitor.cpp
    src/simulation.cpp
    src/bindings.cpp
)

# 创建 Python 模块
pybind11_add_module(optics ${SOURCES})

# 编译选项
if(MSVC)
    # MSVC 编译选项
    target_compile_options(optics PRIVATE 
        /O2           # 优化
        /utf-8        # UTF-8 源文件编码
        /W3           # 警告级别
        /MP           # 多进程编译
    )
    
    # 检查 OpenMP
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_compile_options(optics PRIVATE /openmp)
        target_link_libraries(optics PRIVATE OpenMP::OpenMP_CXX)
    endif()
else()
    # GCC/Clang 编译选项
    target_compile_options(optics PRIVATE 
        -O3           # 最高优化
        -Wall         # 所有警告
        -Wextra       # 额外警告
    )
    
    # 检查 OpenMP
    find_package(OpenMP)
    if(OpenMP_CXX_FOUND)
        target_compile_options(optics PRIVATE -fopenmp)
        target_link_libraries(optics PRIVATE OpenMP::OpenMP_CXX)
    endif()
endif()

# 链接 Python 库
target_link_libraries(optics PRIVATE Python3::Python)

# 设置输出目录
set_target_properties(optics PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
    RUNTIME_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}
)

# Windows 特定：设置输出名称
if(WIN32)
    set_target_properties(optics PROPERTIES
        PREFIX ""
        OUTPUT_NAME "optics"
        SUFFIX ".pyd"
    )
endif()

# 安装规则
install(TARGETS optics DESTINATION lib)

# 打印配置信息
message(STATUS "")
message(STATUS "Optics FDTD Simulator Configuration:")
message(STATUS "  Version: ${PROJECT_VERSION}")
message(STATUS "  C++ Standard: ${CMAKE_CXX_STANDARD}")
message(STATUS "  Python: ${Python3_EXECUTABLE}")
message(STATUS "  pybind11: ${pybind11_VERSION}")
message(STATUS "  OpenMP: ${OpenMP_CXX_FOUND}")
message(STATUS "  Build Type: ${CMAKE_BUILD_TYPE}")
message(STATUS "")