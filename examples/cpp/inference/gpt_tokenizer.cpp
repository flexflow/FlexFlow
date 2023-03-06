#include "gpt_tokenizer.h"

#include <string>

int main(int argc, char * argv[]) {
    char *vocab_file = "/Users/goliaro/Desktop/openai_tokenizer/vocab.bpe";
    char *merge_file = "/Users/goliaro/Desktop/openai_tokenizer/encoder.json";
    
    GPT_Tokenizer tokenizer(merge_file, vocab_file);
    
    std::string line;
    std::vector<std::string> lines;
    std::ifstream infile("/Users/goliaro/Desktop/wikitext-103-raw/wiki.valid.raw");
    if (!infile) {
        std::cout << "Error opening input file" << std::endl;
        return -1;
    }
    std::ofstream outfile("/Users/goliaro/Desktop/wikitext-103-raw/wiki.valid.bpe3", std::ofstream::out);
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
        std::string stripped_line = (line.length() > 0) ? tokenizer.strip(*l) : (*l);
        if (stripped_line.length() == 0) {
            outfile << *l << std::endl;
        } else {
            std::cout << "stripped_line: '" << stripped_line << "' len: " << stripped_line.length() << std::endl;
            tokenizer.padding_encode_single_with_special_tokens(stripped_line, stripped_line.length(), &input_ids, &mask_ids);
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
