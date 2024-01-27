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

#ifndef TRANSCRIBER__QFORMER_TEXT_ENCODER_COMPONENT_HPP_
#define TRANSCRIBER__QFORMER_TEXT_ENCODER_COMPONENT_HPP_

#include <torch/script.h>

#include <bert_tokenizer/tokenizer.hpp>
#include <rclcpp/rclcpp.hpp>
#include <torch_util/type_adapter.hpp>
#include <transcriber/tensor_cache.hpp>

namespace transcriber
{
class QFormerTextEncoder
{
public:
  explicit QFormerTextEncoder(const bool is_cuda, const rclcpp::Logger & logger);
  const bool is_cuda;
  torch::Tensor encode(const std::string & text) const;
  void clearCache();

private:
  const bert_tokenizer::FullTokenizer tokenizer_;
  mutable torch::jit::script::Module model_;
  torch::Tensor tokenize(const std::string & text) const;
  size_t getNumberOfTokens(const std::string & text) const;
  mutable TensorCache<std::string> cache_;
  const rclcpp::Logger logger_;
};
}  // namespace transcriber

#endif  // TRANSCRIBER__QFORMER_TEXT_ENCODER_COMPONENT_HPP_
