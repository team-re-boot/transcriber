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
#include <torch_util/type_adapter.hpp>
#include <transcriber/qformer_text_encoder.hpp>

namespace transcriber
{
QFormerTextEncoder::QFormerTextEncoder(const bool is_cuda)
: is_cuda(is_cuda),
  tokenizer_(get_vocab_path(bert_tokenizer::PretrainedVocab::BERT_BASE_UNCASED)),
  model_(torch::jit::load(
    ament_index_cpp::get_package_share_directory("transcriber") + "/models/qformer_text_encoder.pt",
    (is_cuda && torch::cuda::is_available()) ? torch::kCUDA : torch::kCPU))
{
}

torch::Tensor QFormerTextEncoder::encode(const std::string & text)
{
  return model_
    .forward(
      {tokenize(text).unsqueeze(0),
       torch::ones({1, static_cast<int64_t>(getNumberOfTokens(text))}, at::kInt)})
    .toTensor();
}

torch::Tensor QFormerTextEncoder::tokenize(const std::string & text) const
{
  const auto token_ids = [&]() {
    std::vector<int64_t> ret;
    const auto ids = tokenizer_.convertTokensToIds(tokenizer_.tokenize(text));
    std::transform(ids.begin(), ids.end(), std::back_inserter(ret), [](int id) {
      return static_cast<int64_t>(id);
    });
    return ret;
  }();
  return torch_util::to_torch_tensor(
    torch_msgs::build<torch_msgs::msg::INT64Tensor>().is_cuda(is_cuda).data(token_ids).shape(
      {static_cast<int64_t>(token_ids.size())}));
}

size_t QFormerTextEncoder::getNumberOfTokens(const std::string & text) const
{
  return tokenizer_.convertTokensToIds(tokenizer_.tokenize(text)).size();
}
}  // namespace transcriber
