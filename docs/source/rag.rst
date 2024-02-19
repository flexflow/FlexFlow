:tocdepth: 1
********
RAG Q&A
********

Retrieval Augmented Generation (RAG) combines language models with external knowledge. This use case integrates RAG with FlexFlow Serve for Q&A with documents.

Requirements
============

- FlexFlow Serve setup.
- Retriever setup for RAG.

Implementation
==============

1. FlexFlow Initialization
   Initialize and configure FlexFlow Serve.

2. Data Retrieval Setup
   Setup a retriever for sourcing information relevant to user queries.

3. RAG Integration
   Integrate the retriever with FlexFlow Serve.

4. Response Generation
   Use the LLM with RAG to generate responses based on model's knowledge and retrieved information.

5. Shutdown
   The FlexFlow server automatically shuts down after generating the response.

Example
=======

A complete code example for a web-document Q&A using FlexFlow can be found here: 

1. `Rag Q&A Example with incremental decoding <https://github.com/flexflow/FlexFlow/blob/inference/inference/python/usecases/rag_incr.py>`__

2. `Rag Q&A Example with speculative inference <https://github.com/flexflow/FlexFlow/blob/inference/inference/python/usecases/rag_specinfer.py>`__


Example Implementation:

   .. code-block:: python

      # imports

      # compile and start server
      ff_llm = FlexFlowLLM(...)
      gen_config = ff.GenerationConfig(...)
      ff_llm.compile_and_start(...)
      ff_llm_wrapper = FF_LLM_wrapper(flexflow_llm=ff_llm)
      
      
      # Load web page content
      loader = WebBaseLoader("https://example.com/data")
      data = loader.load()

      # Split text
      text_splitter = RecursiveCharacterTextSplitter(...)
      all_splits = text_splitter.split_documents(data)

      # Initialize embeddings
      embeddings = OpenAIEmbeddings(...) 
      
      # Create VectorStore
      vectorstore = Chroma.from_documents(all_splits, embeddings)
      
      # Use VectorStore as a retriever
      retriever = vectorstore.as_retriever()

      # Apply similarity search 
      question = "Example Question"
      docs = vectorstore.similarity_search(question)
      max_chars_per_doc = 100
      docs_text = ''.join([docs[i].page_content[:max_chars_per_doc] for i in range(len(docs))])
         
      # Using a Prompt Template
      prompt_rag = PromptTemplate.from_template(
         "Summarize the main themes in these retrieved docs: {docs_text}"
      )
      
      # Build Chain
      llm_chain_rag = LLMChain(llm=ff_llm_wrapper, prompt=prompt_rag)

      # Run
      rag_result = llm_chain_rag(docs_text)

      # Stop the server
      ff_llm.stop_server()