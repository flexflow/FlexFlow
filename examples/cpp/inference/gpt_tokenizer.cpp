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

#include "gpt_tokenizer.h"

#include <string>

int main(int argc, char *argv[]) {
  std::string vocab_file = "./gpt2_bpe/vocab.bpe";
  std::string merge_file = "./gpt2_bpe/encoder.json";

  GPT_Tokenizer tokenizer(merge_file, vocab_file);

  std::string line;
  std::vector<std::string> lines;
  std::ifstream infile("./wikitext-103-raw/wiki.valid.raw");
  if (!infile) {
    std::cout << "Error opening input file" << std::endl;
    return -1;
  }
  std::ofstream outfile("./wikitext-103-raw/wiki.valid.bpe.flexflow",
                        std::ofstream::out);
  if (!outfile) {
    std::cout << "Error opening output file" << std::endl;
    return -1;
  }
  while (std::getline(infile, line)) {
    lines.push_back(line);
  }

  std::vector<int64_t> input_ids;
  std::vector<int64_t> mask_ids;
  for (auto l = lines.begin(); l != lines.end(); ++l) {
    std::string stripped_line = tokenizer.strip(*l);
    if (stripped_line.length() == 0) {
      outfile << *l << std::endl;
    } else {
      tokenizer.encode(
          stripped_line, stripped_line.length(), &input_ids, &mask_ids);
      bool first = true;
      for (std::size_t i = 0; i < input_ids.size(); ++i) {
        if (mask_ids[i]) {
          if (!first) {
            outfile << " ";
          } else {
            first = false;
          }
          outfile << input_ids[i];
        }
      }
      outfile << std::endl;
      input_ids.clear();
      mask_ids.clear();
    }
  }
}
