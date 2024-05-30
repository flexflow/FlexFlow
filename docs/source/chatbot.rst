:tocdepth: 1
********
Chatbot
********

The chatbot use case involves setting up a conversational AI model using FlexFlow Serve, capable of engaging in interactive dialogues with users.

Requirements
============

- FlexFlow Serve setup with required configurations.
- Gradio or any interactive interface tool.

Implementation
==============

1. FlexFlow Initialization
   Initialize FlexFlow Serve with desired configurations and specific LLM model.

2. Gradio Interface Setup
   Define a function for response generation based on user inputs. Setup Gradio Chat Interface for interaction. 

   .. code-block:: python
      
      def generate_response(user_input):
         result = llm.generate(user_input)
         return result.output_text.decode('utf-8')
      

3. Running the Interface
   Launch the Gradio interface and interact with the model by entering text inputs.

   .. image:: /imgs/gradio_interface.png
      :alt: Gradio Chatbot Interface
      :align: center

4. Shutdown
   Stop the FlexFlow server after interaction.

Example
=======

Complete code example can be found here: 

1. `Chatbot Example with incremental decoding <https://github.com/flexflow/FlexFlow/blob/inference/inference/python/usecases/gradio_incr.py>`__

2. `Chatbot Example with speculative inference <https://github.com/flexflow/FlexFlow/blob/inference/inference/python/usecases/gradio_specinfer.py>`__


Example Implementation:

   .. code-block:: python

      import gradio as gr
      import flexflow.serve as ff

      ff.init(num_gpus=2, memory_per_gpu=14000, ...)

      def generate_response(user_input):
         result = llm.generate(user_input)
         return result.output_text.decode('utf-8')

      iface = gr.ChatInterface(fn=generate_response)
      iface.launch()