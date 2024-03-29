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

#ifndef TRANSCRIBER__QFORMER_IMAGE_ENCODER_COMPONENT_HPP_
#define TRANSCRIBER__QFORMER_IMAGE_ENCODER_COMPONENT_HPP_

#include <torch/script.h>

#include <rclcpp/rclcpp.hpp>
#include <torch_util/type_adapter.hpp>

namespace transcriber
{
class QFormerImageEncoder
{
public:
  explicit QFormerImageEncoder(const bool is_cuda, const rclcpp::Logger & logger);
  const bool is_cuda;
  torch::Tensor encode(const cv::Mat & image) const;

private:
  mutable torch::jit::script::Module model_;
  const rclcpp::Logger logger_;
};
}  // namespace transcriber

#endif  // TRANSCRIBER__QFORMER_IMAGE_ENCODER_COMPONENT_HPP_
