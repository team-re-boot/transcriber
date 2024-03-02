// Copyright 2024 Team Re-Boot. All rights reserved.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//     http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include <gtest/gtest.h>

#include <opencv2/opencv.hpp>
#include <transcriber/qformer_image_encoder.hpp>
#include <transcriber/qformer_text_encoder.hpp>

// TEST(TextEncoder, encode)
// {
//   auto encoder = transcriber::QFormerTextEncoder(false, rclcpp::get_logger("text_encoder"));
//   encoder.encode("Hello World");
//   // std::cout << encoder.encode("Hello World") << std::endl;
// }

TEST(ImageEncoder, encode)
{
  auto encoder = transcriber::QFormerImageEncoder(false, rclcpp::get_logger("image_encoder"));
  encoder.encode(cv::imread(
    ament_index_cpp::get_package_share_directory("transcriber") + "/data/merlion_demo.png"));
}

int main(int argc, char ** argv)
{
  torch::set_num_threads(32);
  torch::set_num_interop_threads(12);
  std::cout << at::get_parallel_info() << std::endl;
  testing::InitGoogleTest(&argc, argv);
  return RUN_ALL_TESTS();
}
