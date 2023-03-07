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
