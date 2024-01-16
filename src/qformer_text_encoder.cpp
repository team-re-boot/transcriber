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
#include <transcriber/qformer_text_encoder.hpp>

namespace transcriber
{
QFormerTextEncoder::QFormerTextEncoder()
: tokenizer_(get_vocab_path(bert_tokenizer::PretrainedVocab::BERT_BASE_UNCASED)),
  model_(torch::jit::load(
    ament_index_cpp::get_package_share_directory("transcriber") +
    "/models/qformer_text_encoder.pt"))
{
}
}  // namespace transcriber