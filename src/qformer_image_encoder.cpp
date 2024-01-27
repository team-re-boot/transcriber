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

#include <ament_index_cpp/get_package_share_directory.hpp>
#include <transcriber/qformer_image_encoder.hpp>

namespace transcriber
{
QFormerImageEncoder::QFormerImageEncoder(const bool is_cuda, const rclcpp::Logger & logger)
: is_cuda(is_cuda),
  model_((torch::jit::load(
    ament_index_cpp::get_package_share_directory("transcriber") +
      "/models/qformer_image_encoder.pt",
    (is_cuda && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU))),
  logger_(logger)
{
  model_.eval();
  model_ = torch::jit::optimize_for_inference(model_);
}

torch::Tensor QFormerImageEncoder::encode(const cv::Mat & image) const
{
  const auto begin = std::chrono::system_clock::now();
  cv::Mat image_resized;
  cv::resize(image, image_resized, cv::Size(224, 224), cv::INTER_LINEAR);
  /// @todo implement https://pytorch.org/vision/main/generated/torchvision.transforms.Normalize.html

  const auto input_tensor = torch::clamp(
    [&](const auto image) {
      return image.dtype() == torch::kFloat32 ? image : image.to(torch::kFloat32);
    }(torch_util::to_torch_tensor(image_resized)) /
      255.0,
    0.0, 1.0);
  // std::cout << input_tensor.sizes()[0] << std::endl;
  const auto output = model_.forward({input_tensor}).toTensor();
  const auto end = std::chrono::system_clock::now();
  RCLCPP_INFO_STREAM(
    logger_,
    "Elapsed " << std::chrono::duration_cast<std::chrono::milliseconds>(end - begin).count()
               << " ms for running image encoder.");
  return output;
}
}  // namespace transcriber
