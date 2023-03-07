#! /usr/bin/env bash
set -x
set -e

cleanup() {
	rm -rf wikitext-103-raw-v1.zip wikitext-103-raw gpt2_bpe gpt_tokenizer ytokenizer.py bpe.py
}

# Cd into directory holding this script
cd "${BASH_SOURCE[0]%/*}"

# Clean up before test (just in case)
cleanup

# Compile the FlexFlow C++ tokenizer stand-alone
g++ -std=c++11 -I../../../deps/json/include -o gpt_tokenizer gpt_tokenizer.cpp gpt_tokenizer.cc
chmod +x gpt_tokenizer

# Download and inflate wikitext dataset
wget https://s3.amazonaws.com/research.metamind.io/wikitext/wikitext-103-raw-v1.zip
unzip wikitext-103-raw-v1.zip
rm wikitext-103-raw-v1.zip

# Download GPT-2 BPE vocab and merges files
mkdir -p gpt2_bpe
wget -O gpt2_bpe/encoder.json https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/encoder.json
wget -O gpt2_bpe/vocab.bpe https://dl.fbaipublicfiles.com/fairseq/gpt2_bpe/vocab.bpe

# Download minGPT bpe tokenizer for comparison
wget -O bpe.py https://raw.githubusercontent.com/karpathy/minGPT/master/mingpt/bpe.py
chmod +x bpe.py

# Run the FlexFlow C++ tokenizer
./gpt_tokenizer

# Run the minGPT tokenizer
cat << EOF > pytokenizer.py
#!/usr/bin/env python
from bpe import BPETokenizer

tokenizer = BPETokenizer()
inp="./wikitext-103-raw/wiki.valid.raw"
outp="./wikitext-103-raw/wiki.valid.bpe.minGPT"
with open(inp, "r") as infile:
    with open(outp, "w+") as outfile:
        for l in infile.readlines():
            if len(l.strip()) == 0:
                outfile.write(l)
            else:
                out = tokenizer(l.strip()).tolist()[0]
                out = [str(x) for x in out]
                out = " ".join(out)
                outfile.write(out)
                outfile.write("\n")
EOF
chmod +x pytokenizer.py
./pytokenizer.py

# Check that the outputs match
diff ./wikitext-103-raw/wiki.valid.bpe.flexflow ./wikitext-103-raw/wiki.valid.bpe.minGPT

# Clean up after test
cleanup
