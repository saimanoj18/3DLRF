cmake_minimum_required(VERSION 2.6 FATAL_ERROR)


project(3DLRF)
SET(CMAKE_BUILD_TYPE RelWithDebInfo)

find_package(OpenMP)
if (OPENMP_FOUND)
    set (CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
    set (CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
endif()

#find_package(OpenMP REQUIRED)
#set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")



# Adding PCL library
set(PCL_DIR "/home/sai/workspace/pcl-pcl-1.7.2/build/")
find_package(PCL 1.7.2 REQUIRED)
include_directories(${PCL_INCLUDE_DIRS} include)
link_directories(${PCL_LIBRARY_DIRS})
add_definitions(${PCL_DEFINITIONS})




########################################################

add_executable (3DLRF_testing src/3DLRF_testing.cpp)
target_link_libraries (3DLRF_testing ${PCL_LIBRARIES})

add_executable (3DLRF_RRR src/3DLRF_RRR.cpp)
target_link_libraries (3DLRF_RRR ${PCL_LIBRARIES})

add_executable (3DLRF_visualization src/3DLRF_visualization.cpp)
target_link_libraries (3DLRF_visualization ${PCL_LIBRARIES})

add_executable (rops_low_dim src/rops_low_dim.cpp)
target_link_libraries (rops_low_dim ${PCL_LIBRARIES})

add_executable (shot_low_dim src/shot_low_dim.cpp)
target_link_libraries (shot_low_dim ${PCL_LIBRARIES})

add_executable (fpfh_low_dim src/fpfh_low_dim.cpp)
target_link_libraries (fpfh_low_dim ${PCL_LIBRARIES})
