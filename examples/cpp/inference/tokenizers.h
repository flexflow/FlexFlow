/* Copyright 2023 CMU, Facebook, LANL, MIT, NVIDIA, and Stanford (alphabetical)
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

#pragma once
#include <sentencepiece_processor.h>

namespace FlexFlow {

/*!
 * \brief a universal tokenizer that loads
 * either HF's tokenizer or sentence piece, depending on the type.
 */

class Tokenizer {
 public:
  // bos token
  int32_t bos_token_id{1};
  // eos token id
  int32_t eos_token_id{2};

  virtual ~Tokenizer() {}
  virtual std::vector<int32_t> Encode(const std::string& text) = 0;
  virtual std::string Decode(const std::vector<int32_t>& ids) = 0;

  //static std::unique_ptr<Tokenizer> FromFile(const std::string& path);
  //static std::unique_ptr<Tokenizer> ByteLevelBPEFromFile(const std::string& path);
};

class SentencePieceTokenizer : public Tokenizer {
 public:
  SentencePieceTokenizer(const std::string& path) { sentence_piece_.Load(path); }

  std::vector<int32_t> Encode(const std::string& text) final {
    std::vector<int32_t> tokens;
    sentence_piece_.Encode(text, &tokens).IgnoreError();
    return tokens;
  }

  std::string Decode(const std::vector<int32_t>& ids) final {
    std::string text;
    sentence_piece_.Decode(ids, &text).IgnoreError();
    return text;
  }

 private:
  // the tokenizer
  sentencepiece::SentencePieceProcessor sentence_piece_;
};

}; //namespace FlexFlow
