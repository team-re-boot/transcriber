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

#ifndef TRANSCRIBER__TENSOR_CACHE_HPP_
#define TRANSCRIBER__TENSOR_CACHE_HPP_

#include <opencv2/img_hash.hpp>
#include <torch_util/type_adapter.hpp>
#include <tuple>
#include <unordered_map>

namespace detail
{
inline std::size_t hash_mix(std::size_t x) noexcept
{
  constexpr std::uint64_t m = 0xe9846af9b1a615d;
  x ^= x >> 32;
  x *= m;
  x ^= x >> 32;
  x *= m;
  x ^= x >> 28;
  return x;
}

template <class Type>
inline void hash_combine(std::size_t & h, const Type & value) noexcept
{
  h = hash_mix(h + 0x9e3779b9 + std::hash<Type>{}(value));
}
}  // namespace detail

template <>
struct std::hash<std::vector<uint8_t>>
{
  std::size_t operator()(const std::vector<uint8_t> & vector) const
  {
    std::size_t seed = 0;
    for (const auto value : vector) {
      detail::hash_combine(seed, value);
    }
    return seed;
  }
};

namespace transcriber
{
template <typename KeyDataType>
class TensorCache
{
public:
  explicit TensorCache(size_t max_cache_size) : max_cache_size(max_cache_size){};
  const size_t max_cache_size;
  void add(const KeyDataType & tensor) {}

private:
  std::unordered_map<KeyDataType, std::tuple<torch::Tensor, size_t>> data_;
};
}  // namespace transcriber

#endif  // TRANSCRIBER__TENSOR_CACHE_HPP_
