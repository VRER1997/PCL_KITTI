cmake_minimum_required(VERSION 3.13)
project(bin2pcd)

find_package(PCL REQUIRED)

include_directories(${PCL_INCLUDE_DIRS})
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})

add_executable(bin2pcd bin2pcd.cpp)

target_link_libraries (bin2pcd ${PCL_LIBRARIES})