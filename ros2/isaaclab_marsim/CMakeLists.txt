cmake_minimum_required(VERSION 3.16)
project(isaaclab_marsim LANGUAGES C CXX)
set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

find_package(ament_cmake REQUIRED)
find_package(rclcpp REQUIRED)
find_package(geometry_msgs REQUIRED)
find_package(nav_msgs REQUIRED)
find_package(sensor_msgs REQUIRED)
find_package(tf2_ros REQUIRED)
find_package(pcl_conversions REQUIRED)
find_package(mars_quadrotor_msgs REQUIRED)
find_package(marsim_render REQUIRED)
find_package(Eigen3 REQUIRED)
find_package(yaml-cpp REQUIRED)
find_package(OpenCV REQUIRED COMPONENTS core imgproc highgui)
find_package(PCL REQUIRED COMPONENTS common io)
find_package(OpenMP REQUIRED)
find_package(glfw3 REQUIRED)

set(FLIGHTMARE_ROOT "" CACHE PATH "Prepared Flightmare source copy")
set(SUPER_ROOT "" CACHE PATH "Prepared SUPER source copy")
if(NOT EXISTS "${FLIGHTMARE_ROOT}/flightlib/src/objects/quadrotor.cpp")
  message(FATAL_ERROR "Run prepare_flightmare.py and set FLIGHTMARE_ROOT to its output directory")
endif()
set(F "${FLIGHTMARE_ROOT}/flightlib")
add_library(flightlib_dynamics STATIC
  ${F}/src/common/quad_state.cpp ${F}/src/common/command.cpp
  ${F}/src/common/math.cpp ${F}/src/common/logger.cpp
  ${F}/src/common/integrator_base.cpp ${F}/src/common/integrator_rk4.cpp
  ${F}/src/dynamics/quadrotor_dynamics.cpp ${F}/src/dynamics/dynamics_base.cpp
  ${F}/src/objects/quadrotor.cpp ${F}/src/objects/object_base.cpp
  ${F}/src/sensors/imu.cpp ${F}/src/sensors/sensor_base.cpp)
target_include_directories(flightlib_dynamics PUBLIC ${F}/include ${OpenCV_INCLUDE_DIRS})
target_link_libraries(flightlib_dynamics PUBLIC Eigen3::Eigen yaml-cpp ${OpenCV_LIBS})

add_executable(flightmare_node src/flightmare_node.cpp)
target_include_directories(flightmare_node PRIVATE ${SUPER_ROOT}/super_planner/include)
target_link_libraries(flightmare_node flightlib_dynamics)
ament_target_dependencies(flightmare_node rclcpp geometry_msgs nav_msgs tf2_ros mars_quadrotor_msgs)

add_executable(marsim_lidar_node src/marsim_lidar_node.cpp)
# Inline Config resolves PCD paths in the simulator package. The renderer
# library retains its own compiled ROOT_DIR for shaders and scan patterns.
target_compile_definitions(marsim_lidar_node PRIVATE ROOT_DIR="${SUPER_ROOT}/mars_uav_sim/perfect_drone_sim/")
target_include_directories(marsim_lidar_node PRIVATE ${PCL_INCLUDE_DIRS} ${OpenCV_INCLUDE_DIRS})
target_link_libraries(marsim_lidar_node Eigen3::Eigen yaml-cpp glfw ${PCL_LIBRARIES} ${OpenCV_LIBS})
ament_target_dependencies(marsim_lidar_node rclcpp geometry_msgs nav_msgs sensor_msgs pcl_conversions marsim_render)

install(TARGETS flightmare_node marsim_lidar_node DESTINATION lib/${PROJECT_NAME})
install(DIRECTORY launch config DESTINATION share/${PROJECT_NAME})
include(CTest)
if(BUILD_TESTING)
  add_executable(check_flightmare_allocation tests/check_flightmare_allocation.cpp)
  target_link_libraries(check_flightmare_allocation PRIVATE flightlib_dynamics)
  add_test(NAME flightmare_allocation COMMAND check_flightmare_allocation)
  add_executable(replay_position_commands tests/replay_position_commands.cpp)
  target_include_directories(replay_position_commands PRIVATE ${SUPER_ROOT}/super_planner/include)
  target_link_libraries(replay_position_commands PRIVATE flightlib_dynamics)
endif()
ament_package()
