// version 0.1
// Licensed under the MIT License <http://opensource.org/licenses/MIT>.
// SPDX-License-Identifier: MIT
// Copyright (c) 2019-2020 zili wang <wzlnot@gmail.com>.

#include <flexflow/gpt_tokenizer.h>

using json = nlohmann::json;

// codecvt abandoned in c++17
std::wstring GPT_Tokenizer::utf8_to_wstring(std::string const &src) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  return converter.from_bytes(src);
};

std::u32string GPT_Tokenizer::utf8_to_utf32(std::string const &src) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
  return converter.from_bytes(src);
};

std::string GPT_Tokenizer::wstring_to_utf8(std::wstring const &src) {
  std::wstring_convert<std::codecvt_utf8<wchar_t>, wchar_t> converter;
  return converter.to_bytes(src);
};

std::string GPT_Tokenizer::utf32_to_utf8(std::u32string const &src) {
  std::wstring_convert<std::codecvt_utf8<char32_t>, char32_t> converter;
  return converter.to_bytes(src);
};

wchar_t *GPT_Tokenizer::bytes_to_unicode() {
  std::vector<uint32_t> bs;
  for (auto i = uint32_t(L'!'); i < uint32_t(L'~') + 1; ++i) {
    bs.push_back(i);
  }
  for (auto i = uint32_t(L'¡'); i < uint32_t(L'¬') + 1; ++i) {
    bs.push_back(i);
  }
  for (auto i = uint32_t(L'®'); i < uint32_t(L'ÿ') + 1; ++i) {
    bs.push_back(i);
  }
  std::vector<uint32_t> cs = bs;
  uint32_t n = 0;
  for (uint32_t b = 0; b < 256; ++b) {
    auto p = find(bs.begin(), bs.end(), b);
    if (p == bs.end()) {
      bs.push_back(b);
      cs.push_back(256 + n);
      n++;
    }
  }
  static wchar_t bytes_mapping[256] = {};
  for (size_t i = 0; i < 256; i++) {
    bytes_mapping[i] = i;
  }
  for (size_t i = 0; i < bs.size(); i++) {
    bytes_mapping[bs[i]] = cs[i];
  }
  return bytes_mapping;
}

void GPT_Tokenizer::unicode_to_bytes() {
  for (int i = 0; i < 256; i++) {
    bytes_decoder[bytes_encoder[i]] = (char)i;
  }
}

std::vector<std::string> GPT_Tokenizer::split(std::string const &s,
                                              std::regex rgx) {
  std::vector<std::string> elems;
  std::sregex_token_iterator iter(s.begin(), s.end(), rgx, -1);
  std::sregex_token_iterator end;
  while (iter != end) {
    elems.push_back(*iter);
    ++iter;
  }
  return elems;
};

std::string GPT_Tokenizer::strip(std::string const &inpt) {
  if (inpt.length() == 0) {
    return inpt;
  }
  auto start_it = inpt.begin();
  auto end_it = inpt.rbegin();
  while (std::isspace(*start_it)) {
    ++start_it;
  }
  if (start_it == inpt.end()) {
    return "";
  }
  while (std::isspace(*end_it)) {
    ++end_it;
  }
  return std::string(start_it, end_it.base());
}

std::unordered_set<wbigram_pair, hash_pair>
    GPT_Tokenizer::get_pairs(std::vector<std::wstring> word) {
  std::unordered_set<wbigram_pair, hash_pair> pairs;
  std::wstring prev_char = word[0];
  for (size_t i = 1; i < word.size(); ++i) {
    pairs.insert(wbigram_pair({prev_char, word[i]}));
    prev_char = word[i];
  }
  return pairs;
};

void GPT_Tokenizer::load_vocab(std::string const &vocab_file) {
  std::ifstream file_handle(vocab_file);
  assert(file_handle.good() && "file not exists");
  bool discard_first_line = false;
  if (discard_first_line) {
    std::string first_line_discard;
    std::getline(file_handle, first_line_discard); // skip the first line
  }
  json vocab_data_ = json::parse(file_handle,
                                 /*parser_callback_t */ nullptr,
                                 /*allow_exceptions */ true,
                                 /*ignore_comments */ true);
  auto vocab_ = vocab_data_.get<std::unordered_map<std::string, int32_t>>();
  for (auto item : vocab_) {
    vocab.insert({item.first, item.second});
    inverse_vocab.insert({item.second, item.first});
  }
};

void GPT_Tokenizer::load_merge(std::string const &merge_file) {
  bpe_ranks.reserve(60000);
  std::ifstream file_handle(merge_file);
  assert(file_handle.good() && "file not exists");
  std::string line;
  uint32_t curr_idx = 0;
  std::string version_substring = "#version:";
  while (getline(file_handle, line)) {
    if (line.size() == 0 || line.rfind(version_substring, 0) == 0) {
      continue;
    }
    std::vector<std::string> bigrams = split(line);
    assert(bigrams.size() == 2 && "unk format");
    wbigram_pair curr(utf8_to_wstring(bigrams[0]), utf8_to_wstring(bigrams[1]));
    bpe_ranks.insert({curr, curr_idx});
    curr_idx++;
  }
};

std::vector<std::string> GPT_Tokenizer::bpe(std::wstring token) {
  // bpe use wstring
  if (cache.find(token) != cache.end()) {
    return cache[token];
  }
  std::vector<std::wstring> wword;
  for (auto c : token) {
    wword.push_back(std::wstring(1, c));
  }
  std::unordered_set<wbigram_pair, hash_pair> pairs = get_pairs(wword);
  if (pairs.empty()) {
    return {wstring_to_utf8(token)};
  }

  while (true) {
    auto bigram = pairs.begin();
    if (pairs.size() > 1) {
      bigram = std::min_element(
          pairs.begin(),
          pairs.end(),
          [this](wbigram_pair const &a, wbigram_pair const &b) -> bool {
            if (bpe_ranks.find(a) == bpe_ranks.end()) {
              return false;
            }
            if (bpe_ranks.find(b) == bpe_ranks.end()) {
              return true;
            }
            return bpe_ranks[a] < bpe_ranks[b];
          });
    }
    if (bpe_ranks.find(*bigram) == bpe_ranks.end()) {
      break;
    }
    std::wstring first = bigram->first;
    std::wstring second = bigram->second;
    decltype(wword) new_wword;

    auto i = wword.begin();
    while (i < wword.end()) {
      auto j = std::find(i, wword.end(), first);
      if (j == wword.end()) {
        new_wword.insert(new_wword.end(), i, wword.end());
        break;
      }
      new_wword.insert(new_wword.end(), i, j);
      i = j;
      // i <= wword.end
      if (*i == first && i < wword.end() - 1 && *(i + 1) == second) {
        new_wword.push_back(first + second);
        i += 2;
      } else {
        new_wword.push_back(*i);
        i += 1;
      }
    }
    wword = new_wword;
    if (wword.size() == 1) {
      break;
    } else {
      pairs = get_pairs(wword);
    }
  }
  std::vector<std::string> word;
  for (auto w : wword) {
    word.push_back(wstring_to_utf8(w));
  }
  if (token.size() < cache_word_max_length && cache.size() < cache_max_size) {
    cache.insert({token, word});
  }
  return word;
};

std::vector<std::string> GPT_Tokenizer::tokenize(std::string str) {
  std::vector<std::string> bpe_tokens;
  std::wstring wstr = utf8_to_wstring(str);
  std::wsregex_iterator iter(wstr.begin(), wstr.end(), pat);
  std::wsregex_iterator end;
  while (iter != end) {
    std::wstring token;
    for (char c : wstring_to_utf8(iter->str())) {
      if (0 > c) {
        token.push_back(*(bytes_encoder + c + 256));
      } else {
        token.push_back(*(bytes_encoder + c));
      }
    }
    if (token.length() > 0) {
      decltype(bpe_tokens) curr_bpe_tokens = bpe(token);
      bpe_tokens.insert(
          bpe_tokens.end(), curr_bpe_tokens.begin(), curr_bpe_tokens.end());
    }
    ++iter;
  }
  return bpe_tokens;
}

int32_t GPT_Tokenizer::convert_token_to_id(std::string token) {
  auto p = vocab.find(token);
  if (p != vocab.end()) {
    return vocab[token];
  } else {
    return vocab[unk_token];
  }
}

void GPT_Tokenizer::encode(std::string str,
                           size_t max_length,
                           std::vector<int32_t> *input_ids,
                           std::vector<int32_t> *mask_ids) {
  if (not input_ids->empty()) {
    input_ids->clear();
  }
  if (not mask_ids->empty()) {
    mask_ids->clear();
  }
  input_ids->reserve(max_length);
  mask_ids->reserve(max_length);
  // input_ids->push_back(vocab[bos_token]);
  // mask_ids->push_back(1);
  auto tokens = tokenize(str);
  for (auto t : tokens) {
    if (input_ids->size() == max_length - 1) {
      break;
    }
    input_ids->push_back(convert_token_to_id(t));
    mask_ids->push_back(1);
  }
  // input_ids->push_back(vocab[eos_token]);
  // mask_ids->push_back(1);
  while (input_ids->size() < max_length) {
    input_ids->push_back(vocab[pad_token]);
    mask_ids->push_back(0);
  }
  if (mode == OPT_TOKENIZER) {
    mask_ids->insert(mask_ids->begin(), 1);
    input_ids->insert(input_ids->begin(), 2);
  }
}

std::string GPT_Tokenizer::decode(std::vector<int32_t> input_ids,
                                  std::vector<int32_t> mask_ids) {
  // look up each number in encoder.json dictionary
  std::ostringstream oss;
  int index = 0;
  for (auto const &id : input_ids) {
    if (index == 0) {
      if (mode == OPT_TOKENIZER) {
        if (id == 2) {
          index++;
        }
        continue;
      }
    }
    if (!mask_ids[index]) {
      index++;
      continue;
    }
    auto it = inverse_vocab.find(id);
    if (it != inverse_vocab.end()) {
      oss << it->second;
    } else {
      // Handle the case when the integer is not found in the inverse_vocab map.
      // You can choose to ignore it, skip it, or handle it differently based on
      // your requirements.
      assert(false);
    }
    index++;
  }
  std::string concatenated_tokens = oss.str();
  // apply byte_decoder to each character in the input_ids string, then decode
  // as utf-8
  std::wstring wstr = utf8_to_wstring(concatenated_tokens);
  std::string result;
  for (wchar_t ch : wstr) {
    result += bytes_decoder[ch];
  }
  return result;
}
