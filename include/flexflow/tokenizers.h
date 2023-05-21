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
#include "gpt_tokenizer.h"
#include <sentencepiece_processor.h>

namespace FlexFlow {

/*!
 * \brief a universal tokenizer that loads
 * either HF's tokenizer or sentence piece, depending on the type.
 */

class Tokenizer {
public:
  // bos token
  int32_t bos_token_id{0};
  // eos token id
  int32_t eos_token_id{1};

  virtual ~Tokenizer() {}
  virtual std::vector<int32_t> Encode(std::string const &text) = 0;
  virtual std::string Decode(std::vector<int32_t> const &ids) = 0;

  // static std::unique_ptr<Tokenizer> FromFile(const std::string& path);
  // static std::unique_ptr<Tokenizer> ByteLevelBPEFromFile(const std::string&
  // path);
};

class SentencePieceTokenizer : public Tokenizer {
public:
  SentencePieceTokenizer(std::string const &path) {
    sentence_piece_.Load(path);
  }

  std::vector<int32_t> Encode(std::string const &text) final {
    std::vector<int32_t> tokens;
    sentence_piece_.Encode(text, &tokens).IgnoreError();
    return tokens;
  }

  std::string Decode(std::vector<int32_t> const &ids) final {
    std::string text;
    sentence_piece_.Decode(ids, &text).IgnoreError();
    return text;
  }

private:
  // the tokenizer
  sentencepiece::SentencePieceProcessor sentence_piece_;
};

class OptTokenizer : public Tokenizer {
public:
  OptTokenizer(std::string const &vocab_file,  // path to "gpt2-vocab.json"
               std::string const &merges_file) // path to "gpt2-merges.txt"
      : tokenizer(OPT_TOKENIZER, vocab_file, merges_file) {
    bos_token_id = 0;
    eos_token_id = 2;
  }

  std::vector<int32_t> Encode(std::string const &text) final {
    std::vector<int32_t> tokens;
    std::vector<int32_t> mask_ids;
    tokenizer.encode(text, text.length(), &tokens, &mask_ids);

    auto it = std::find(mask_ids.begin(), mask_ids.end(), 0);

    if (it != mask_ids.end()) {
      size_t index = std::distance(mask_ids.begin(), it);
      tokens.erase(tokens.begin() + index, tokens.end());
    }

    return tokens;
  }

  std::string Decode(std::vector<int32_t> const &ids) final {
    std::vector<int32_t> mask_ids;
    for (int i = 0; i < ids.size(); i++) {
      mask_ids.push_back(1);
    }
    std::string text = tokenizer.decode(ids, mask_ids);
    return text;
  }

private:
  GPT_Tokenizer tokenizer;
};

}; // namespace FlexFlow
