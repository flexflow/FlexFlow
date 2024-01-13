:tocdepth: 1
***********************
FlexFlow Serve FastAPI
***********************

Introduction
============

The Python API for FlexFlow Serve enables users to initialize, manage and interact with large language models (LLMs) via FastAPI or Gradio.

Requirements
------------

- FlexFlow Serve setup with necessary configurations.
- FastAPI and Uvicorn for running the API server.

API Configuration
=================

Users can configure the API using FastAPI to handle requests and manage the model.

1. FastAPI Application Initialization
   Initialize the FastAPI application to create API endpoints.

2. Request Model Definition
   Define the model for API requests using Pydantic.

3. Global Variable for LLM Model
   Declare a global variable to store the LLM model.

Example
-------

.. code-block:: python

   from fastapi import FastAPI
   from pydantic import BaseModel
   import flexflow.serve as ff

   app = FastAPI()

   class PromptRequest(BaseModel):
       prompt: str

   llm_model = None

Endpoint Creation
=================

Create API endpoints for LLM interactions to handle generation requests.

1. Initialize Model on Startup
   Use the FastAPI event handler to initialize and compile the LLM model when the API server starts.

2. Generate Response Endpoint
   Create a POST endpoint to generate responses based on the user's prompt.

Example
-------

.. code-block:: python

   @app.on_event("startup")
   async def startup_event():
       global llm_model
       # Initialize and compile the LLM model
       # ...

   @app.post("/generate/")
   async def generate(prompt_request: PromptRequest):
       global llm_model
       if llm_model is None:
           raise HTTPException(status_code=503, detail="LLM model is not initialized.")
       
       with llm_model:
           full_output = llm_model.generate([prompt_request.prompt])[0].output_text.decode('utf-8')
           return {"prompt": prompt_request.prompt, "response": full_output}

Running and Testing
===================

Instructions for running and testing the FastAPI server.

1. Run the FastAPI Server
   Use Uvicorn to run the FastAPI server with specified host and port.

2. Testing the API
   Make requests to the API endpoints and verify the responses.

Example
-------

.. code-block:: bash
    # Running within the inference/python folder:
    uvicorn entrypoint.fastapi_incr:app --reload --port 3000

