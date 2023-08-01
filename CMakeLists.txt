cmake_minimum_required(VERSION 3.10)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

project(sembed VERSION 1.0)
add_executable(sembed ${CMAKE_CURRENT_SOURCE_DIR}/src/main/main.cpp)
add_library(load_from_binary STATIC ${CMAKE_CURRENT_SOURCE_DIR}/src/load/load_from_binary.cpp)
target_include_directories(sembed PUBLIC ${CMAKE_CURRENT_SOURCE_DIR}/src/include)
target_link_libraries(sembed PUBLIC load_from_binary)
