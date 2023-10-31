import sys
sys.path.append('.')
from typing import List, Optional
from models import FF_LLM


from langchain.document_loaders import WebBaseLoader
from langchain.chains.summarize import load_summarize_chain
from langchain.chat_models import ChatOpenAI

# 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

loader = WebBaseLoader("https://blogs.nvidia.com/blog/2023/08/09/generative-ai-auto-industry/?ncid=so-nvsh-653212-vt03")
docs = loader.load()

# openai example
llm = ChatOpenAI(temperature=0, model_name="gpt-3.5-turbo-16k")
# ff
# llm = FF_LLM()
chain = load_summarize_chain(llm, chain_type="stuff")

chain.run(docs)