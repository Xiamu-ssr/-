cmake_minimum_required(VERSION 3.10.0)
# set compiler
set(CMAKE_C_COMPILER "/home/mumu/intel/oneapi/compiler/2022.1.0/linux/bin/intel64/icc")
set(CMAKE_CXX_COMPILER "/home/mumu/intel/oneapi/compiler/2022.1.0/linux/bin/intel64/icpc")
# set(CMAKE_C_COMPILER "gcc")
# set(CMAKE_CXX_COMPILER "g++")

# complier option
# add_definitions("-g -qopt-report-file=report.txt -O2 -ipo -vec -xAVX -parallel -par-num-threads=14 -par-schedule-guided=1 -par-affinity=granularity=fine,verbose,scatter")
add_definitions("-g -O2 -ipo -vec -xAVX -parallel -par-num-threads=14 -par-schedule-guided=1")

# set project name
project(Mmul)

find_package(OpenMP REQUIRED)

include_directories(./include/)

add_executable(main main.cpp src/matrix.cpp)
# add_executable(main main.cpp)
# LINK_DIRECTORIES(./src)
# target_link_libraries(main m)
target_link_libraries(main OpenMP::OpenMP_CXX)

