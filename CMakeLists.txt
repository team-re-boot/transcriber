cmake_minimum_required(VERSION 3.8)
project(transcriber)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake_auto REQUIRED)
find_package(TBB REQUIRED)
ament_auto_find_build_dependencies()

ament_auto_add_library(${PROJECT_NAME} SHARED
  src/qformer_text_encoder.cpp
  src/qformer_image_encoder.cpp
)
# target_link_libraries(${PROJECT_NAME}
#   tbb
# )

file(DOWNLOAD
  https://huggingface.co/hakuturu583/torchscript_lavis/resolve/main/qformer_image_encoder.pt
  qformer_image_encoder.pt
  SHOW_PROGRESS
  EXPECTED_MD5 3d7452ce7565a028e81a3eb2dfcd0ef8)

file(DOWNLOAD
  https://huggingface.co/hakuturu583/torchscript_lavis/resolve/main/qformer_text_encoder.pt
  qformer_text_encoder.pt
  SHOW_PROGRESS
  EXPECTED_MD5 68afcb1f76dfc4a8acf9eb28c5bb20bc)

install(
  FILES ${CMAKE_CURRENT_BINARY_DIR}/qformer_image_encoder.pt ${CMAKE_CURRENT_BINARY_DIR}/qformer_text_encoder.pt
  DESTINATION share/${PROJECT_NAME}/models
)

install(DIRECTORY data
  DESTINATION share/${PROJECT_NAME}
)

if(BUILD_TESTING)
  find_package(ament_lint_auto REQUIRED)
  ament_lint_auto_find_test_dependencies()
  find_package(ament_cmake_gtest REQUIRED)
  ament_add_gtest(test_qformer_encoder test/test_qformer_encoder.cpp)
  target_link_libraries(test_qformer_encoder ${PROJECT_NAME})
endif()

ament_auto_package()
