cmake_minimum_required(VERSION 3.15)
project(optics VERSION 0.1.0 LANGUAGES CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# 查找 Python 和 pybind11
find_package(Python3 REQUIRED COMPONENTS Interpreter Development)
find_package(pybind11 CONFIG REQUIRED)

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
    target_compile_options(optics PRIVATE /O2 /openmp)
else()
    target_compile_options(optics PRIVATE -O3 -fopenmp)
endif()

# 链接库
if(OpenMP_CXX_FOUND)
    target_link_libraries(optics PRIVATE OpenMP::OpenMP_CXX)
endif()

# 安装
install(TARGETS optics DESTINATION lib)