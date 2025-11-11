cmake_minimum_required(VERSION 3.8)
project(simple_3d_localization)
set(PROJECT_DIR simple_3d_localization)
set(PROJECT_NAMESPACE s3l)

# find package
find_package(ament_cmake_auto REQUIRED)
ament_auto_find_build_dependencies()

# compiler settings & policies
cmake_policy(SET CMP0074 NEW)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Option
option(S3L_USE_FAST_GICP "Link fast_gicp" OFF)
option(S3L_USE_SMALL_GICP "Link small_gicp" ON)
option(S3L_DISABLE_EIGEN_ALIGN "Disable Eigen alignment/vectorization" OFF)

# Compiler options
if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
  set(CMAKE_CXX_FLAGS_RELEASE "${CMAKE_CXX_FLAGS_RELEASE} -O3")
  set(CMAKE_CXX_FLAGS_DEBUG "${CMAKE_CXX_FLAGS_DEBUG} -O0 -g")

  if(NOT CMAKE_BUILD_TYPE OR CMAKE_BUILD_TYPE STREQUAL "Release" OR CMAKE_BUILD_TYPE STREQUAL "RelWithDebInfo")
    add_compile_definitions(NDEBUG)
  endif()

  message(STATUS "CMAKE_CXX_FLAGS: ${CMAKE_CXX_FLAGS}")
  message(STATUS "CMAKE_CXX_FLAGS_RELEASE: ${CMAKE_CXX_FLAGS_RELEASE}")
  message(STATUS "CMAKE_CXX_FLAGS_DEBUG: ${CMAKE_CXX_FLAGS_DEBUG}")
endif()

# OpenMP support
find_package(OpenMP)
if(OpenMP_FOUND)
  set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")
  set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
  set(CMAKE_EXE_LINKER_FLAGS "${CMAKE_EXE_LINKER_FLAGS} ${OpenMP_EXE_LINKER_FLAGS}")
  message(STATUS "OpenMP found and enabled.")
else()
  message(WARNING "OpenMP not found. Parallel processing will be disabled.")
endif()

# External libraries
find_package(PCL 1.12 REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(Boost REQUIRED COMPONENTS system thread date_time)
find_package(small_gicp REQUIRED)
find_package(Sophus REQUIRED)
find_package(ndt_omp REQUIRED)
find_package(fast_gicp REQUIRED)
find_package(Ceres REQUIRED)

# ROS IDL generation
set(msg_files
  "msg/ScanMatchingStatus.msg"
)
set(srv_files
  "srv/QueryGlobalLocalization.srv"
  "srv/SetGlobalMap.srv"
)
rosidl_generate_interfaces(${PROJECT_NAME}
  ${msg_files}
  ${srv_files}
  DEPENDENCIES std_msgs geometry_msgs sensor_msgs
)


file(GLOB_RECURSE globalmap_server_src_files
  LIST_DIRECTORIES false
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "src/${PROJECT_DIR}/globalmap_server_node.cpp")

file(GLOB_RECURSE hdl_localization_src_files
  LIST_DIRECTORIES false
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "src/${PROJECT_DIR}/hdl_localization/node.cpp")

file(GLOB_RECURSE imu_initializer_src_files
  LIST_DIRECTORIES false
  RELATIVE "${CMAKE_CURRENT_SOURCE_DIR}"
  "src/${PROJECT_DIR}/utils/imu_initializer_node.cpp")

# Globalmap Server
list(LENGTH globalmap_server_src_files globalmap_src_count)
if(globalmap_src_count GREATER 0)
  ament_auto_add_library(globalmap_server_node SHARED ${globalmap_server_src_files})

  target_include_directories(globalmap_server_node PUBLIC ${PCL_INCLUDE_DIRS})
  target_link_libraries(globalmap_server_node ${PCL_LIBRARIES})

  rclcpp_components_register_nodes(globalmap_server_node
    "${PROJECT_NAMESPACE}::map::GlobalmapServerNode"
  )
endif()

# HDL Localization
list(LENGTH hdl_localization_src_files hdl_src_count)
if(hdl_src_count GREATER 0)
  ament_auto_add_library(hdl_localization_node SHARED ${hdl_localization_src_files})

  rosidl_get_typesupport_target(cpp_typesupport_target "${PROJECT_NAME}" "rosidl_typesupport_cpp")
  target_link_libraries(hdl_localization_node "${cpp_typesupport_target}")

  target_include_directories(hdl_localization_node PUBLIC
    ${PCL_INCLUDE_DIRS}
    ${Eigen3_INCLUDE_DIRS}
    $<$<BOOL:${S3L_USE_SMALL_GICP}>:${small_gicp_INCLUDE_DIRS}>
    ${ndt_omp_INCLUDE_DIRS}
    $<$<BOOL:${S3L_USE_FAST_GICP}>:${fast_gicp_INCLUDE_DIRS}>
    $<$<NOT:$<TARGET_EXISTS:Sophus::Sophus>>:${Sophus_INCLUDE_DIRS}>
  )

  target_link_libraries(hdl_localization_node
    ${PCL_LIBRARIES}
    Eigen3::Eigen
    $<$<BOOL:${S3L_USE_SMALL_GICP}>:${small_gicp_LIBRARIES}>
    ${ndt_omp_LIBRARIES}
    $<$<BOOL:${S3L_USE_FAST_GICP}>:${fast_gicp_LIBRARIES}>
    $<$<TARGET_EXISTS:Sophus::Sophus>:Sophus::Sophus>
  )

  if(S3L_DISABLE_EIGEN_ALIGN)
    target_compile_definitions(hdl_localization_node PRIVATE
      EIGEN_DONT_ALIGN_STATICALLY=1
      EIGEN_MAX_ALIGN_BYTES=0
      EIGEN_DONT_VECTORIZE=1
    )
  endif()
  
  rclcpp_components_register_nodes(hdl_localization_node
    "${PROJECT_NAMESPACE}::hdl_localization::LocalizationNode"
  )
endif()

# IMU Initializer
list(LENGTH imu_initializer_src_files imu_src_count)
if(imu_src_count GREATER 0)
  ament_auto_add_library(imu_initializer_node SHARED ${imu_initializer_src_files})
  target_link_libraries(imu_initializer_node Ceres::ceres)
  ament_target_dependencies(imu_initializer_node rclcpp rclcpp_components)
  rclcpp_components_register_nodes(imu_initializer_node
    "${PROJECT_NAMESPACE}::utils::ImuInitializerNode"
  )
endif()
ament_export_dependencies(Ceres)
set(CMAKE_INSTALL_RPATH_USE_LINK_PATH TRUE)

foreach(target globalmap_server_node hdl_localization_node imu_initializer_node)
  if(TARGET ${target})
    target_compile_features(${target} PUBLIC cxx_std_17)
    target_compile_options(${target} PRIVATE -Wall -Wextra -Wpedantic)
  endif()
endforeach()

install(DIRECTORY
  launch
  rviz
  config
  DESTINATION share/${PROJECT_NAME}/)

if(BUILD_TESTING)
  ament_auto_find_test_dependencies()
  ament_lint_auto_find_test_dependencies()
endif()

ament_auto_package()
