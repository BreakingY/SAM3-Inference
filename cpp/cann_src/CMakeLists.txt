cmake_minimum_required(VERSION 3.0)
project("sam3-infer")
set(CMAKE_CXX_STANDARD 17)
add_compile_options(-g  -fpermissive)
include_directories(/usr/local/include /usr/local/include/opencv4
                    /usr/local/Ascend/ascend-toolkit/latest/runtime/include
                    /usr/local/Ascend/ascend-toolkit/latest/runtime/include/acl
                    /usr/local/Ascend/ascend-toolkit/latest/runtime/include/acl/dvpp)
include_directories(./)

add_definitions(-DENABLE_DVPP_INTERFACE)
link_directories(/usr/local/Ascend/ascend-toolkit/latest/runtime/lib64 /usr/local/lib)


set(EXECUTABLE_OUTPUT_PATH ./)
add_compile_options(-g  -fpermissive) 
set(CMAKE_BUILD_TYPE Release)

add_executable(sam3_pcs_app ./sam3_cann_infer.cpp)
target_link_libraries(sam3_pcs_app ascendcl acl_dvpp_mpi opencv_core opencv_imgproc opencv_imgcodecs opencv_highgui opencv_videoio opencv_video)


