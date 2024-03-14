:tocdepth: 1
****************
Prompt Template
****************

Prompt templates guide the model's response generation. This use case demonstrates setting up FlexFlow Serve to integrate with Langchain and using prompt templates to handle dynamic prompt templates.

Requirements
============

- FlexFlow Serve setup with appropriate configurations.
- Langchain integration with templates for prompt management.

Implementation
==============

1. FlexFlow Initialization
   Initialize and configure FlexFlow Serve.

2. LLM Setup
   Compile and start the server for text generation.

3. Prompt Template Setup
   Setup a prompt template for guiding model's responses.

4. Response Generation
   Use the LLM with the prompt template to generate a response.

5. Shutdown
   Stop the FlexFlow server after generating the response.

Example
=======

Complete code example can be found here: 

1. `Prompt Template Example with incremental decoding <https://github.com/flexflow/FlexFlow/blob/inference/inference/python/usecases/prompt_template_incr.py>`__

2. `Prompt Template Example with speculative inference <https://github.com/flexflow/FlexFlow/blob/inference/inference/python/usecases/prompt_template_specinfer.py>`__


Example Implementation:

   .. code-block:: python

      import flexflow.serve as ff
      from langchain.prompts import PromptTemplate

      ff_llm = FlexFlowLLM(...)
      ff_llm.compile_and_start(...)

      template = "Question: {question}\nAnswer:"
      prompt = PromptTemplate(template=template, input_variables=["question"])

      response = ff_llm.generate("Who was the US president in 1997?")
