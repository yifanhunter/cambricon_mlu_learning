cmake_minimum_required(VERSION 3.5)

set(CMAKE_CXX_FLAGS "-Werror -g -fPIC -std=c++11 -pthread")
add_definitions(-D_GLIBCXX_USE_CXX11_ABI=${ABI})

project(op_amp)
INCLUDE_DIRECTORIES($ENV{NEUWARE_HOME}/include 
                    ${PROJECT_SOURCE_DIR}/include)
LINK_DIRECTORIES($ENV{NEUWARE_HOME}/lib64)

add_executable(op_amp 
    ${PROJECT_SOURCE_DIR}/sample_api/sample_model.cc)
target_link_libraries(op_amp  cnrt magicmind_runtime magicmind)
